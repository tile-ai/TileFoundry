"""Validate full-width Qwen models under real NCCL 1F1B training."""

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path

from config import FULL, SOURCES, TINY


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", choices=("full", "tiny"), default="full")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1, help="expert partition of the data axis")
    parser.add_argument("--packed", action="store_true", help="multiple documents with unequal padding per data lane")
    parser.add_argument("--routing-check", choices=("strict", "replay-near-ties"), default="strict",
                        help="validation only: optionally replay oracle choices after verifying near-tie margins")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "dmuon"), default="sgd")
    parser.add_argument("--checkpoint", choices=("auto", "stage", "none"), default="auto")
    parser.add_argument("--boundary", choices=("stage", "layer"), default="layer")
    parser.add_argument("--overlap", choices=("off", "on"), default="off")
    parser.add_argument("--ns-backend", choices=("direct", "gram"), default="direct")
    parser.add_argument("--forward-unshard", choices=("broadcast", "all_gather"), default="broadcast",
                        help="attention matrix publication; explicit expert layouts require broadcast in the pin")
    parser.add_argument("--muon-max-norm", type=float,
                        help="clip the global Muon-only norm; AdamW parameters are excluded")
    parser.add_argument("--replicas", type=int, default=1, help="HSDP replica dimension inside each stage's DP group")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="run this many unprofiled timing steps, then this many profiled steps")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--reference-offload", action="store_true",
                        help="keep only stage-owned reference gradients and offload the reference between checks")
    parser.add_argument("--compare-final", type=Path,
                        help="require bitwise final weights matching a prior run with the same data steps")
    parser.add_argument("--transport", choices=("native", "socket"), default="native")
    parser.add_argument("--output", type=Path, default=Path(".tmp/qwen3-dmuon/pipeline"))
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 0))
    if args.stages < 2 or args.layers < args.stages or args.layers % args.stages:
        parser.error("layers must be a positive multiple of at least two pipeline stages")
    if not world or world % args.stages:
        parser.error("launch with torchrun; world size must be a multiple of stages")
    if min(args.cp, args.ep) < 1 or (world // args.stages) % (args.cp * args.ep):
        parser.error("CP times EP must divide ranks per pipeline stage")
    args.packed = args.packed or args.cp > 1 or args.ep > 1
    if args.packed and (args.tokens < 32 or args.tokens % args.cp):
        parser.error("packed sequences need >= 32 tokens divisible by CP")
    if args.packed and (args.optimizer != "dmuon" or args.replicas != 1 or args.muon_max_norm is not None
                        or args.boundary != "layer"):
        parser.error("packed CP/EP currently requires dmuon, layer boundaries, no HSDP replicas or norm clipping")
    if args.replicas < 1 or (world // args.stages) % args.replicas:
        parser.error("HSDP replicas must divide the DP group size")
    if args.replicas > 1 and args.forward_unshard == "all_gather":
        parser.error("pinned dmuon exports stale replica-row weights after HSDP all-gather publication; use broadcast")
    if args.profile_steps < 0 or args.warmup_steps < 0:
        parser.error("profile and warmup step counts must be nonnegative")
    if args.muon_max_norm is not None and not 0 < args.muon_max_norm < float("inf"):
        parser.error("Muon max norm must be finite and positive")
    if args.optimizer != "dmuon" and (args.overlap == "on" or args.replicas > 1):
        parser.error("HSDP and optimizer overlap require --optimizer dmuon")
    if args.optimizer != "dmuon" and (args.muon_max_norm is not None or args.ns_backend != "direct"
                                     or args.forward_unshard != "broadcast"):
        parser.error("Muon options require --optimizer dmuon")
    if args.compare_final and (args.optimizer != "dmuon" or args.compare_final.resolve() == args.output.resolve()):
        parser.error("final-state comparison needs dmuon and a different baseline output directory")
    if args.microbatches < args.stages or args.tokens < 2 or args.steps < 1:
        parser.error("require microbatches >= stages, tokens >= 2, and steps >= 1")
    return args


def main():
    args = arguments()
    if args.transport == "socket":
        os.environ.update(NCCL_NET="Socket", NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1",
                          NCCL_NVLS_ENABLE="0", NCCL_MNNVL_ENABLE="0")
    args.output.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if (args.output / f"rank{rank}.json").exists():
        raise FileExistsError("output contains completed results; select a fresh directory")
    os.environ.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,NET,P2P,COLL,ENV",
                      NCCL_DEBUG_FILE=str(args.output.resolve() / f"nccl-rank{rank}.log"),
                      NCCL_TOPO_DUMP_FILE=str(args.output.resolve() / f"topology-rank{rank}.xml"))
    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining import Schedule1F1B
    from transformers import Qwen3MoeForCausalLM
    import transformers

    from tilefoundry.ir import types
    from pipeline_model import QwenStage, pipeline_loss
    from pipeline_trace import TracedStage, check_endpoints

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    replicas = world // args.stages
    index, replica = divmod(rank, replicas)
    pp_group = dp_group = None
    for data_rank in range(replicas):
        group = dist.new_group([stage * replicas + data_rank for stage in range(args.stages)])
        if data_rank == replica:
            pp_group = group
    for stage in range(args.stages):
        group = dist.new_group(list(range(stage * replicas, (stage + 1) * replicas)))
        if stage == index:
            dp_group = group
    replica_group = shard_group = None
    shards = replicas // args.replicas
    if args.replicas > 1:
        for stage_index in range(args.stages):
            for row in range(args.replicas):
                ranks = [stage_index * replicas + row * shards + column for column in range(shards)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    shard_group = group
            for column in range(shards):
                ranks = [stage_index * replicas + row * shards + column for row in range(args.replicas)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    replica_group = group
    shape = replace(FULL if args.width == "full" else TINY, layers=args.layers)
    if shape.experts % args.ep:
        raise ValueError("EP must divide the expert count")
    parallel = None
    if args.packed:
        from packed import Pack, Parallel
        from parallel_optimizer import ParallelOptimizer

        parallel = Parallel(args.stages, args.cp, args.ep)
    torch.manual_seed(1234)
    config = shape.transformers_config()
    if args.packed:
        config._attn_implementation = "sdpa"
    with torch.device(device):
        reference = Qwen3MoeForCausalLM(config)
    assert sum(p.numel() for p in reference.parameters()) == shape.parameter_count()
    local_model = QwenStage(reference, index, args.stages).to(device)
    local_model.routing_check = args.routing_check
    if parallel:
        parallel.apply(local_model)
    owned = dict(local_model.named_parameters())
    owned_parameters = sum(p.numel() for p in owned.values())
    reference_params = dict(reference.named_parameters())
    if args.reference_offload:
        for name, parameter in reference_params.items():
            parameter.requires_grad_(name in owned)
    if args.optimizer == "dmuon":
        import dedicated

        ranks = torch.arange(index * replicas, (index + 1) * replicas).reshape(args.replicas, shards)
        if parallel:
            optimizer = ParallelOptimizer(local_model, parallel, dp_group, shape, overlap=args.overlap == "on",
                                           ns_backend=args.ns_backend, forward_unshard=args.forward_unshard)
        else:
            optimizer = dedicated.configure(local_model, dp_group, shape, overlap=args.overlap == "on",
                                             boundary=args.boundary, replicate_group=replica_group,
                                             shard_group=shard_group, ranks=ranks, ns_backend=args.ns_backend,
                                             forward_unshard=args.forward_unshard)
        owned = dict.fromkeys(owned)
    else:
        optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
    checkpoint = args.checkpoint == "stage" or (args.checkpoint == "auto" and args.optimizer == "dmuon")
    if checkpoint:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        stage_model = checkpoint_wrapper(local_model)
    else:
        stage_model = local_model
    mesh = types.Mesh((types.Topology("gpu", world),), types.Layout((args.stages, replicas)), ("pp", "dp"))
    local_tokens = args.tokens // args.cp
    hidden = torch.empty((1, local_tokens, shape.hidden), device=device)
    routers = torch.empty((1, local_model.end, local_tokens, shape.experts), device=device)
    inputs = (torch.empty((1, local_tokens), dtype=torch.long, device=device),) if index == 0 else (
        hidden, torch.empty((1, local_model.begin, local_tokens, shape.experts), device=device))
    outputs = (torch.empty((1, local_tokens, shape.vocab), device=device) if local_model.last else hidden, routers)
    for value in (*inputs, *outputs):
        if value.is_floating_point():
            value.requires_grad_(True)
    stage = TracedStage(stage_model, index, args.stages, device, input_args=inputs, output_args=outputs,
                        input_grads=tuple(torch.empty_like(x) if x.is_floating_point() else None for x in inputs),
                        output_grads=tuple(torch.empty_like(x) for x in outputs), group=pp_group,
                        mesh=mesh[:, replica], shape=shape, tokens=local_tokens)
    loss_fn = (lambda outputs, targets: local_model.pack.loss(outputs, targets, local_model.config)) if parallel else partial(
        pipeline_loss, config=local_model.config, microbatches=args.microbatches)
    schedule = Schedule1F1B(stage, args.microbatches, loss_fn=loss_fn, scale_grads=False)
    stage.semantic_model = local_model
    packs = []

    def batch(step):
        data_rank = parallel.data_rank if parallel else replica
        generator = torch.Generator().manual_seed(9000 + 100 * step + data_rank)
        full = torch.randint(shape.vocab, (args.microbatches, args.tokens), generator=generator).to(device)
        if parallel:
            local_model.pack = Pack(parallel, args.tokens, args.microbatches, step, device, dp_group)
            packs.append(dict(step=step, **local_model.pack.summary))
            return full, local_model.pack.local(full), local_model.pack.local(local_model.pack.targets(full))
        return full, full, full

    def snapshot():
        return optimizer.snapshot() if parallel else dedicated.snapshot(local_model, dp_group)

    reports = []
    for step in range(args.steps):
        published = None
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        if args.reference_offload:
            reference.to(device)
            reference_params = dict(reference.named_parameters())
        stage.iteration = step
        full_tokens, tokens, targets = batch(step)
        handles = []
        if parallel:
            local_model.route_oracle = {name: [] for name in local_model.model.layers}
            for name in local_model.model.layers:
                def save_routes(module, inputs, output, name=name):
                    local_model.route_oracle[name].append((output[2].detach().reshape(1, args.tokens, shape.top_k),
                                                          output[0].detach().reshape(1, args.tokens, shape.experts)))
                handles.append(reference.model.layers[int(name)].mlp.gate.register_forward_hook(save_routes))
        expected_losses = []
        for microbatch in full_tokens.split(1):
            if parallel:
                loss = local_model.pack.reference(reference, microbatch)
                output = None
            else:
                output = reference(input_ids=microbatch, labels=microbatch, use_cache=False)
                loss = output.loss / args.microbatches
            expected_losses.append(loss.detach() * args.microbatches)
            loss.backward()
        for handle in handles:
            handle.remove()
        if args.reference_offload:
            reference.cpu()
            reference_params = dict(reference.named_parameters())
            torch.cuda.empty_cache()
        losses = []
        schedule.step(*(tokens,) if index == 0 else (), target=targets if local_model.last else None,
                      losses=losses if local_model.last else None, return_outputs=False)
        max_gradient_error = 0.0
        checked = 0
        for name, parameter in owned.items() if args.optimizer == "sgd" else ():
            expected = reference_params[name].grad
            assert (parameter.grad is None) == (expected is None), name
            if expected is None:
                continue
            actual = parameter.grad
            expected = expected.to(actual.device)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6, msg=lambda m: f"{name}: {m}")
            max_gradient_error = max(max_gradient_error, (actual - expected).abs().max().item())
            checked += 1
            if replicas > 1:
                dist.all_reduce(actual, op=dist.ReduceOp.AVG, group=dp_group)
        if args.optimizer == "dmuon":
            checked, max_gradient_error = optimizer.check_gradients(reference_params) if parallel else dedicated.check_gradients(
                local_model, reference_params, owned, dp_group)
        if local_model.last:
            if parallel:
                combined = torch.stack(losses)
                dist.all_reduce(combined, op=dist.ReduceOp.AVG, group=parallel.cp_group)
                losses = list(combined.unbind())
            torch.testing.assert_close(torch.stack(losses) * args.microbatches,
                                       torch.stack(expected_losses), rtol=1e-5, atol=1e-6)
        clipping = None
        if args.muon_max_norm is not None:
            clipping = dedicated.check_clipping(optimizer, reference_params, owned, pp_group, args.muon_max_norm)
        optimizer.step()
        published = snapshot() if args.optimizer == "dmuon" else owned
        with torch.no_grad():
            for owner_stage in range(args.stages):
                for name, parameter in reference.named_parameters():
                    if name.startswith("model.layers."):
                        owner = int(name.split(".")[2]) // (args.layers // args.stages)
                    else:
                        owner = 0 if name == "model.embed_tokens.weight" else args.stages - 1
                    if owner != owner_stage:
                        continue
                    if args.reference_offload:
                        payload = published[name] if owner == index else torch.empty_like(parameter, device=device)
                        dist.broadcast(payload, src=owner * replicas + replica, group=pp_group)
                        parameter.copy_(payload)
                        del payload
                    else:
                        if owner == index:
                            parameter.copy_(published[name])
                        dist.broadcast(parameter, src=owner * replicas + replica, group=pp_group)
        reports.append(dict(step=step, checked_gradient_units=checked, max_gradient_error=max_gradient_error,
                            clipping=clipping,
                            reference_loss=torch.stack(expected_losses).mean().item(),
                            pipeline_loss=torch.stack(losses).mean().item() * args.microbatches if losses else None))
    validation_peak = torch.cuda.max_memory_allocated(device)
    profile = None
    total_steps = args.steps
    if args.profile_steps:
        from profiling import capture

        reference.zero_grad(set_to_none=True)
        reference_params.clear()
        del reference, output, parameter, published, loss
        torch.cuda.empty_cache()
        training_losses = []
        if parallel:
            parallel.record = False
            local_model.route_oracle = None

        def training_step(step):
            stage.iteration = step
            _, tokens, targets = batch(step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            schedule.step(*(tokens,) if index == 0 else (), target=targets if local_model.last else None,
                          losses=losses if local_model.last else None, return_outputs=False)
            if args.optimizer == "sgd" and replicas > 1:
                for parameter in local_model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=dp_group)
            with torch.profiler.record_function("training/optimizer"):
                if args.muon_max_norm is not None:
                    dedicated.dmuon.clip_grad_norm_(optimizer, args.muon_max_norm)
                optimizer.step()
            if losses:
                training_losses.append(torch.stack(losses).mean().detach() * args.microbatches)

        profile = capture(training_step, first=args.steps, warmup=args.warmup_steps, steps=args.profile_steps,
                          output=args.output, rank=rank)
        if training_losses:
            values = torch.stack(training_losses)
            assert torch.isfinite(values).all(), "nonfinite training loss"
            profile["losses"] = values.tolist()
        if args.optimizer == "dmuon":
            published = snapshot()
        total_steps += args.warmup_steps + 2 * args.profile_steps
    final_weights = None
    if args.optimizer == "dmuon":
        final_weights = dedicated.fingerprints(published)
    if args.compare_final:
        baseline = json.loads((args.compare_final / f"rank{rank}.json").read_text())
        assert baseline["shape"] == asdict(shape), "replay model shape differs"
        assert (baseline.get("cp", 1), baseline.get("ep", 1), baseline.get("packed", False)) == (
            args.cp, args.ep, args.packed), "replay parallelism or packing differs"
        assert (baseline["pp"], baseline["dp"], baseline["microbatches"], baseline["tokens"],
                baseline["total_training_steps"]) == (
                    args.stages, replicas, args.microbatches, args.tokens, total_steps), "replay inputs differ"
        assert final_weights and final_weights == baseline["final_weight_sha256"], "final weight replay differs"
    gathered = [None] * world
    dist.all_gather_object(gathered, (os.uname().nodename, stage.transfers))
    hosts = sorted(set(host for host, _ in gathered))
    matched = check_endpoints([events for _, events in gathered],
                             total_steps * args.microbatches * (args.stages - 1) * replicas * 4)
    for step in range(total_steps):
        execution = [event for event in stage.execution if event["step"] == step]
        for action in ("forward", "backward"):
            assert [e["microbatch"] for e in execution if e["action"] == action] == list(range(args.microbatches))
        if index == 0 and args.microbatches > args.stages:
            first_backward = next(i for i, e in enumerate(execution) if e["action"] == "backward")
            last_forward = max(i for i, e in enumerate(execution) if e["action"] == "forward")
            assert first_backward < last_forward, "expected interleaved forward/backward ordering"
    result = dict(status="passed", shape=asdict(shape), parameters=shape.parameter_count(),
                  rank=rank, stage=index, data_rank=parallel.data_rank if parallel else replica,
                  stage_local_rank=replica, pp=args.stages, dp=replicas,
                  cp=args.cp, ep=args.ep, packed=args.packed, packing=packs,
                  cp_rank=parallel.cp_rank if parallel else 0, ep_rank=parallel.ep_rank if parallel else 0,
                  expert_gradient_group_size=dist.get_world_size(parallel.expert_group) if parallel else replicas,
                  routing_check=args.routing_check, routing_comparisons=local_model.routing_comparisons,
                  local_tokens=local_tokens, data_lanes=parallel.data_size if parallel else replicas,
                  cp_ep_contracts=list(parallel.contracts.values()) if parallel else [],
                  microbatches=args.microbatches, tokens=args.tokens, steps=reports,
                  total_training_steps=total_steps, final_weight_sha256=final_weights,
                  compared_final=str(args.compare_final) if args.compare_final else None,
                  optimizer=args.optimizer, activation_checkpointing=checkpoint,
                  ns_backend=args.ns_backend, forward_unshard=args.forward_unshard,
                  forward_unshard_scope="attention matrices; packed expert MatrixLayout uses owner broadcast",
                  muon_max_norm=args.muon_max_norm,
                  boundary=args.boundary, overlap=args.overlap, hsdp_replicas=args.replicas, hsdp_shards=shards,
                  forward_prefetch_depth=int(args.overlap == "on"),
                  post_step_prefetch_groups=int(args.overlap == "on" and args.replicas == 1),
                  dmuon_version=dedicated.dmuon.__version__ if args.optimizer == "dmuon" else None,
                  loss="global valid-target mean plus valid-target-weighted per-pack HF router auxiliary loss" if parallel
                  else "mean of per-microbatch HF causal loss including cross-layer router auxiliary loss",
                  schedule="torch.distributed.pipelining.Schedule1F1B", sources=SOURCES,
                  torch=torch.__version__, transformers=transformers.__version__, nccl=torch.cuda.nccl.version(),
                  dtype="float32", transport=args.transport, physical_hosts=len(hosts),
                  logical_stage_groups=[list(range(i * replicas, (i + 1) * replicas)) for i in range(args.stages)],
                  emulation="all P2P/SHM/NVLS disabled; Socket transport" if args.transport == "socket" else None,
                  owned_parameters=owned_parameters,
                  peak_allocated_bytes=validation_peak, profile=profile,
                  reference_offload=args.reference_offload,
                  reference_gradient_scope="stage-owned parameters" if args.reference_offload else "all parameters",
                  memory_scope="peak across reference and training phases; reference offloaded between checks"
                  if args.reference_offload else "includes full unpartitioned oracle, stage copy and their gradients",
                  matched_transfers=matched, execution=stage.execution, transfers=stage.transfers,
                  reference_communication=["updated stage weights broadcast to full-model oracles after every step",
                                           "reference gradients averaged across DP peers; dmuon snapshots compared across peers"],
                  communication_model_scope="PP Exchange and observed CP/EP Collective/Exchange contracts; runtime counts specialized",
                  remaining=["uncheckpointed dmuon pipeline drain", "TP composition with CP/EP/PP and full training HIR",
                             "HSDP sharded-publication checkpoint consistency and sharded explicit expert layouts",
                             "optimizer checkpoint/resume across differently sized PP stages and full-model clipping",
                             "mixed compute/communication scheduling and transport cost realization"])
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({k: v for k, v in result.items()
                          if k not in {"execution", "transfers", "final_weight_sha256", "cp_ep_contracts",
                                       "routing_comparisons"}}, indent=2))


if __name__ == "__main__":
    main()
