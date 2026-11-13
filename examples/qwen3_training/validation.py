"""Independent numerical reference and version-pinned DMuon conformance checks."""

import hashlib

import torch
import torch.distributed as dist

from tilefoundry import utils

dmuon = utils.LazyModule("dmuon")


def tolerances():
    """BF16 attention rounds each CP shard before the FP32 gradient reduction.

    Bound both norm error and maximum error by two BF16 epsilons. Scale the
    latter by the matrix maximum because reductions can cancel individual values.
    Retain the FP32 absolute floor and loss check; the norm check protects small
    gradients that the absolute floor alone would admit.
    This is an explicit mixed-precision contract, not bitwise equivalence.
    """
    from .attention import DTYPE

    relative = 2 * torch.finfo(DTYPE).eps
    return dict(
        gradient_rtol=relative,
        gradient_atol=1e-6,
        gradient_scale="matrix_max",
        gradient_relative_l2=relative,
        routing_rtol=relative,
        routing_atol=1e-7,
        loss_rtol=1e-5,
        loss_atol=1e-6,
    )


def gradient_error(actual, expected, policy, name):
    expected = expected.to(actual.device)
    absolute_tolerance = policy["gradient_atol"] + policy["gradient_rtol"] * expected.abs().max().item()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=absolute_tolerance,
        msg=lambda m: f"{name}: {m}",
    )
    difference = (actual - expected).double()
    relative = (
        difference.norm() / expected.double().norm().clamp_min(torch.finfo(torch.float64).tiny)
    ).item()
    if relative > policy["gradient_relative_l2"]:
        raise AssertionError(
            f"{name}: relative gradient L2 error {relative} exceeds {policy['gradient_relative_l2']}"
        )
    return difference.abs().max().item(), relative


def check_gradients(model, reference, names, group, policy, *, expert_slice=None):
    """Read pinned dmuon owner buffers and compare normalized accumulated gradients."""
    dmuon.prepare_muon_grads(model)
    for name in names:
        if dist.get_world_size(group) > 1 and reference[name].grad is not None:
            gradient = reference[name].grad
            payload = gradient.to(device=torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(payload, op=dist.ReduceOp.AVG, group=group)
            if payload is not gradient:
                gradient.copy_(payload)
    modules = {id(module): name for name, module in model.named_modules()}
    checked, error, relative = 0, 0.0, 0.0
    for unit in dmuon.get_dedicated_params(model):
        if not unit.is_owner:
            continue
        name = getattr(unit, "_param_fqn", None)
        if name is None:
            prefix = modules[id(unit.module)]
            name = f"{prefix}.{unit.param_name}" if prefix else unit.param_name
        expected = reference[name].grad
        if expert_slice is not None and ".experts." in name:
            expected = expected[expert_slice]
        spec = getattr(unit, "matrix_spec", None)
        if spec is not None:
            expected = expected.flatten()[spec.offset : spec.offset + spec.numel].view(spec.shape)
        actual = unit._reduced_grad
        assert actual is not None, f"missing owner gradient: {name}"
        absolute_error, relative_error = gradient_error(actual, expected, policy, name)
        error, relative = max(error, absolute_error), max(relative, relative_error)
        checked += 1
    return checked, error, relative


def check_publication(weights, group):
    """Require identical complete stage weights on every data peer."""
    if dist.get_world_size(group) > 1:
        root = dist.get_global_rank(group, 0)
        for name, weight in weights.items():
            expected = weight.clone()
            dist.broadcast(expected, src=root, group=group)
            torch.testing.assert_close(
                weight, expected, rtol=0, atol=0, msg=lambda m: f"{name}: {m}"
            )


def optimizer_gradients(optimizer, reference, policy):
    results = [
        check_gradients(
            model,
            reference,
            names,
            optimizer.dense_group,
            policy,
            expert_slice=optimizer.expert_slice if i else None,
        )
        for i, (model, names) in enumerate(zip(optimizer.models, optimizer.names))
    ]
    return (
        sum(row[0] for row in results),
        max(row[1] for row in results),
        max(row[2] for row in results),
    )


def reference_backward(reference, stage, inputs, microbatches):
    """Evaluate the independent full model, retaining only reference gradients."""
    reference.zero_grad(set_to_none=True)
    reference.to(inputs.device)
    stage.route_oracle = RoutingOracle(stage, reference, inputs.shape[1])
    losses = []
    try:
        for microbatch in inputs.split(1):
            loss = stage.pack.reference(reference, microbatch)
            losses.append(loss.detach() * microbatches)
            loss.backward()
    finally:
        stage.route_oracle.close()
    reference.cpu()
    torch.cuda.empty_cache()
    return losses


def check_loss(stage, losses, expected, groups, microbatches):
    if stage.last:
        combined = torch.stack(losses)
        dist.all_reduce(combined, op=dist.ReduceOp.AVG, group=groups.cp_group)
        losses = list(combined.unbind())
        policy = tolerances()
        torch.testing.assert_close(
            torch.stack(losses) * microbatches,
            torch.stack(expected),
            rtol=policy["loss_rtol"],
            atol=policy["loss_atol"],
        )
    return losses


@torch.no_grad()
def publish_reference(reference, published, groups, device, stages):
    """Advance the reference from each stage's actual update, outside timed work."""
    for owner_stage in range(stages):
        for name, parameter in reference.named_parameters():
            if name.startswith("model.layers."):
                owner = int(name.split(".")[2]) // (reference.config.num_hidden_layers // stages)
            else:
                owner = 0 if name == "model.embed_tokens.weight" else stages - 1
            if owner != owner_stage:
                continue
            source = owner * groups.size + groups.local_rank
            payload = published[name] if owner == groups.stage else torch.empty_like(parameter, device=device)
            dist.broadcast(payload, src=source, group=groups.pipeline)
            parameter.copy_(payload)


def fingerprints(weights):
    """Hash complete stage weights outside timing windows for deterministic replay."""
    return {
        name: hashlib.sha256(weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, weight in weights.items()
    }


def weight_samples(weights):
    """Small deterministic diagnostics; gradient checks still cover all owner values."""
    result = {}
    for name, weight in weights.items():
        if not torch.isfinite(weight).all():
            raise ValueError(f"nonfinite final weights: {name}")
        result[name] = (
            weight.detach().flatten()[:: max(1, weight.numel() // 256)][:256].cpu().tolist()
        )
    return result


class RoutingOracle:
    """Check probabilities and replay only choices inside the rounding margin."""

    def __init__(self, stage, reference, tokens):
        self.routes = {name: [] for name in stage.model.layers}
        self.handles = []
        for name in stage.model.layers:

            def save(module, inputs, output, name=name):
                self.routes[name].append(
                    (
                        output[2].detach().reshape(1, tokens, stage.config.num_experts_per_tok),
                        output[0].detach().reshape(1, tokens, stage.config.num_experts),
                    )
                )

            self.handles.append(
                reference.model.layers[int(name)].mlp.gate.register_forward_hook(save)
            )

    def close(self):
        for handle in self.handles:
            handle.remove()

    def check(self, stage, name, output, batch, tokens):
        expected, probabilities = self.routes[name][stage.microbatch]
        actual = output[2].reshape(batch, tokens, -1)
        expected, probabilities = stage.pack.local(expected), stage.pack.local(probabilities)
        actual_probabilities = output[0].reshape(batch, tokens, -1)
        policy = tolerances()
        torch.testing.assert_close(
            actual_probabilities,
            probabilities,
            atol=policy["routing_atol"],
            rtol=policy["routing_rtol"],
        )
        mismatch = (actual.sort(dim=-1).values != expected.sort(dim=-1).values).any(-1)
        delta = (actual_probabilities - probabilities).abs().amax(-1)
        top = probabilities.topk(stage.config.num_experts_per_tok + 1).values
        margin = top[..., -2] - top[..., -1]
        ambiguous = margin <= 2 * delta + 2 * torch.finfo(probabilities.dtype).eps * top[..., 0]
        if mismatch.any() and (mismatch & ~ambiguous).any():
            raise AssertionError(
                f"layer {name} microbatch {stage.microbatch}: "
                f"{mismatch.sum().item()} tokens changed selected expert set; "
                f"{(mismatch & ~ambiguous).sum().item()} exceed rounding bound"
            )
        stage.routing_comparisons.append(
            dict(
                layer=name,
                microbatch=stage.microbatch,
                changed_sets=mismatch.sum().item(),
                max_probability_error=delta.max().item(),
            )
        )
        indices = expected.reshape(-1, stage.config.num_experts_per_tok)
        weights = output[0].gather(-1, indices)
        return output[0], weights / weights.sum(-1, keepdim=True), indices
