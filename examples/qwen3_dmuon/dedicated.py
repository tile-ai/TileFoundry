"""Stage-local dmuon policy and pinned-version owner-gradient conformance checks."""

import hashlib

import dmuon
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


def configure(model, group, shape, *, overlap=False, boundary="layer", replicate_group=None,
              shard_group=None, ranks=None, ns_backend="direct", forward_unshard="broadcast", boundaries=None):
    """Dedicate attention matrices and individual expert projections within one stage."""
    if replicate_group is None:
        mesh = DeviceMesh.from_group(group, "cuda", mesh_dim_names=("shard",))
        shard_mesh, replicate_mesh = mesh, None
    else:
        mesh = DeviceMesh.from_group([replicate_group, shard_group], "cuda", mesh=ranks,
                                     mesh_dim_names=("replicate", "shard"))
        shard_mesh, replicate_mesh = mesh["shard"], mesh["replicate"]
    if boundaries is None:
        boundaries = set(model.model.layers.values())
        if model.first:
            boundaries.add(model.model.embed_tokens)
        if model.last:
            boundaries.update((model.model.norm, model.lm_head))
        if boundary == "stage":
            boundaries = {model}

    def policy(name, parameter):
        if name.endswith("experts.gate_up_proj"):
            return dict(route="muon", matrix_layout=dmuon.MatrixLayout.view(
                parameter.shape[0], 2, shape.expert_hidden, shape.hidden))
        if name.endswith("experts.down_proj"):
            return dict(route="muon", matrix_layout=dmuon.MatrixLayout.view(
                parameter.shape[0], shape.hidden, shape.expert_hidden))
        if ".self_attn." in name and "_proj.weight" in name:
            return dict(route="muon", muon_forward_unshard=forward_unshard)
        return dict(route="adamw")

    dmuon.dedicate_params(model, shard_mesh, replicate_mesh=replicate_mesh,
                         predicate=lambda name, p: p.requires_grad,
                         hook_boundary_predicate=lambda module: module in boundaries,
                         param_policy_fn=policy, process_group_policy="isolated")
    return dmuon.Muon(model, ns_backend=ns_backend, replicate_async=overlap,
                      forward_prefetch_depth=1 if overlap else 0,
                      post_step_prefetch_groups=int(overlap and replicate_group is None),
                      first_step_progress_log=False)


def check_gradients(model, reference, names, group, *, expert_slice=None):
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
    checked, error = 0, 0.0
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
            expected = expected.flatten()[spec.offset:spec.offset + spec.numel].view(spec.shape)
        actual = unit._reduced_grad
        assert actual is not None, f"missing owner gradient: {name}"
        expected = expected.to(actual.device)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6, msg=lambda m: f"{name}: {m}")
        error = max(error, (actual - expected).abs().max().item())
        checked += 1
    return checked, error


def snapshot(model, group):
    """Drain publication and require identical complete stage weights on DP peers."""
    dmuon.wait_all_post_step_broadcasts(model)
    torch.cuda.synchronize()
    weights = dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=False)
    if dist.get_world_size(group) > 1:
        root = dist.get_global_rank(group, 0)
        for name, weight in weights.items():
            expected = weight.clone()
            dist.broadcast(expected, src=root, group=group)
            torch.testing.assert_close(weight, expected, rtol=0, atol=0, msg=lambda m: f"{name}: {m}")
    return weights


def fingerprints(weights):
    """Hash complete stage weights outside timing windows for deterministic replay."""
    return {name: hashlib.sha256(weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, weight in weights.items()}


def weight_samples(weights):
    """Small deterministic diagnostics; gradient checks still cover all owner values."""
    result = {}
    for name, weight in weights.items():
        if not torch.isfinite(weight).all():
            raise ValueError(f"nonfinite final weights: {name}")
        result[name] = weight.detach().flatten()[::max(1, weight.numel() // 256)][:256].cpu().tolist()
    return result


def check_clipping(optimizer, reference, names, pp_group, max_norm):
    """Check dmuon's Muon-only global norm against one copy of each stage's matrices."""
    matrices = [reference[name].grad for name in names
                if (".self_attn." in name and "_proj.weight" in name)
                or name.endswith(("experts.gate_up_proj", "experts.down_proj"))]
    squared = torch.stack([gradient.double().square().sum() for gradient in matrices]).sum()
    squared = squared.to(device=torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(squared, group=pp_group)
    expected = squared.sqrt().float()
    stats = dmuon.clip_grad_norm_(optimizer, max_norm)
    torch.testing.assert_close(stats.total_norm, expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(stats.clip_coef, (max_norm / (expected + 1e-6)).clamp(max=1),
                               rtol=1e-5, atol=1e-7)
    return dict(scope="Muon matrices only; AdamW parameters excluded",
                total_norm=stats.total_norm.item(), clip_coef=stats.clip_coef.item())
