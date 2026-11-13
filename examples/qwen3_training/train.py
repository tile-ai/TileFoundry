"""Run or compare Qwen training with one shared execution and validation path."""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

from .model import FULL, SOURCES, TINY

CASES = {
    "cp": dict(cp=2, ep=1, replicas=1),
    "cp_ep": dict(cp=2, ep=2, replicas=1),
    "cp_hsdp": dict(cp=2, ep=1, replicas=2),
    "ep_hsdp": dict(cp=1, ep=2, replicas=2),
}


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", choices=("full", "tiny"), default="full")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1, help="HSDP replica dimension within each stage")
    parser.add_argument("--overlap", choices=("off", "on"), default="off")
    parser.add_argument("--steps", type=int, default=2, help="reference-checked updates")
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--timing-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def validate_arguments(args, world):
    if args.stages < 2 or args.layers < args.stages or args.layers % args.stages:
        raise ValueError("layers must be a positive multiple of at least two pipeline stages")
    if not world or world % args.stages:
        raise ValueError("world size must be a multiple of pipeline stages")
    if min(args.cp, args.ep) < 1 or (world // args.stages) % (args.cp * args.ep):
        raise ValueError("CP times EP must divide ranks per pipeline stage")
    if args.tokens < 32 or args.tokens % args.cp:
        raise ValueError("packed sequences need >=32 positions divisible by CP")
    if args.replicas < 1 or (world // (args.stages * args.ep)) % args.replicas:
        raise ValueError("HSDP replicas must divide both dense and expert parameter domains")
    if args.profile_steps < 0 or args.warmup_steps < 0 or min(args.timing_rounds,args.timing_steps,args.steps) < 1:
        raise ValueError("profile/warmup counts must be nonnegative; update and timing counts must be positive")
    if args.microbatches < args.stages:
        raise ValueError("microbatches must cover the pipeline stages")


def run(argv):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    from tilefoundry.ir import types

    local_rank = types.Placement.from_env().ids["gpu"]
    import torch
    import torch.distributed as dist

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    validate_arguments(args, world)
    if (args.output / f"rank{rank}.json").exists():
        raise FileExistsError("output contains completed results; select a fresh directory")
    import transformers
    from torch.distributed.pipelining import Schedule1F1B
    from transformers import Qwen3MoeForCausalLM

    from . import sharding, validation
    from .data import Pack
    from .model import QwenStage
    from .parallel import Parallel
    from .trace import TracedStage, capture, check_endpoints

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    groups = Parallel(args.stages, args.cp, args.ep, args.replicas)
    index, replica, replicas = groups.stage, groups.local_rank, groups.size
    pp_group, dp_group = groups.pipeline, groups.data
    shards = groups.shards
    shape = replace(FULL if args.width == "full" else TINY, layers=args.layers)
    if shape.experts % args.ep:
        raise ValueError("EP must divide the expert count")
    parallel = groups
    torch.manual_seed(1234)
    config = shape.transformers_config()
    with torch.device(device):
        reference = Qwen3MoeForCausalLM(config)
    assert sum(p.numel() for p in reference.parameters()) == shape.parameter_count()
    local_model = QwenStage(reference, index, args.stages).to(device)
    parallel.apply(local_model)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from . import attention

    ALL_ATTENTION_FUNCTIONS.register("qwen3_reference", attention.reference)
    for layer in reference.model.layers:
        layer.self_attn.config = copy.copy(layer.self_attn.config)
        layer.self_attn.config._attn_implementation = "qwen3_reference"
        layer.self_attn.get_pack = lambda: local_model.pack
    owned = dict(local_model.named_parameters())
    owned_parameters = sum(p.numel() for p in owned.values())
    for name, parameter in reference.named_parameters():
        parameter.requires_grad_(name in owned)
    del parameter
    optimizer = sharding.Optimizer(local_model, groups, overlap=args.overlap == "on")
    del owned
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    stage_model = checkpoint_wrapper(local_model)
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
                        mesh=groups.mesh[:, groups.outer_rank, groups.ep_rank, groups.cp_rank], shape=shape, tokens=local_tokens)
    loss_fn = lambda outputs, targets: local_model.pack.loss(outputs, targets, local_model.config)
    schedule = Schedule1F1B(stage, args.microbatches, loss_fn=loss_fn, scale_grads=False)
    stage.semantic_model = local_model
    packs = []

    def batch(step):
        data_rank = parallel.data_rank
        generator = torch.Generator().manual_seed(9000 + 100 * step + data_rank)
        full = torch.randint(shape.vocab, (args.microbatches, args.tokens), generator=generator).to(device)
        local_model.pack = Pack(parallel, args.tokens, args.microbatches, step, device, dp_group)
        packs.append(dict(step=step, **local_model.pack.summary))
        return full, local_model.pack.local(full), local_model.pack.local(local_model.pack.targets(full))

    def backward(tokens, targets):
        losses = []
        schedule.step(*(tokens,) if index == 0 else (), target=targets if local_model.last else None,
                      losses=losses if local_model.last else None, return_outputs=False)
        return losses

    reports = []
    numerical_policy = validation.tolerances()
    for step in range(args.steps):
        published = None
        optimizer.zero_grad(set_to_none=True)
        stage.iteration = step
        full_tokens, tokens, targets = batch(step)
        expected_losses = validation.reference_backward(reference, local_model, full_tokens,
                                                        args.microbatches)
        reference_params = dict(reference.named_parameters())
        losses = backward(tokens, targets)
        checked, max_gradient_error, max_gradient_relative_l2 = validation.optimizer_gradients(
            optimizer, reference_params, numerical_policy)
        losses = validation.check_loss(local_model, losses, expected_losses, groups, args.microbatches)
        optimizer.step()
        published = optimizer.snapshot()
        validation.check_publication(published, dp_group)
        validation.publish_reference(reference, published, groups, device, args.stages)
        reports.append(dict(step=step, checked_gradient_units=checked, max_gradient_error=max_gradient_error,
                            max_gradient_relative_l2=max_gradient_relative_l2,
                            reference_loss=torch.stack(expected_losses).mean().item(),
                            pipeline_loss=torch.stack(losses).mean().item() * args.microbatches if losses else None))
    validation_peak = torch.cuda.max_memory_allocated(device)
    profile = None
    total_steps = args.steps
    if args.profile_steps:
        reference.zero_grad(set_to_none=True)
        reference_params.clear()
        del reference, published
        torch.cuda.empty_cache()
        training_losses = []
        parallel.record = False
        local_model.route_oracle = None

        def training_step(step):
            stage.iteration = step
            _, tokens, targets = batch(step)
            optimizer.zero_grad(set_to_none=True)
            losses = backward(tokens, targets)
            with torch.profiler.record_function("training/optimizer"):
                optimizer.step()
            if losses:
                training_losses.append(torch.stack(losses).mean().detach() * args.microbatches)

        profile = capture(training_step, first=args.steps, warmup=args.warmup_steps, steps=args.profile_steps,
                          output=args.output, rank=rank, rounds=args.timing_rounds, timing_steps=args.timing_steps)
        measured_packs = [pack for pack in packs if profile["first_timed_step"] <= pack["step"] < profile["first_profiled_step"]]
        valid_targets = sum(pack["valid_targets"] for pack in measured_packs) * args.microbatches
        denominator = torch.tensor(valid_targets, dtype=torch.int64, device=device)
        dist.all_reduce(denominator, group=dp_group)
        profile["valid_targets_per_step"] = denominator.item() / (args.cp * profile["measured_steps"])
        profile["valid_targets_per_second"] = profile["valid_targets_per_step"] / profile["unprofiled_seconds_per_step"]
        if training_losses:
            values = torch.stack(training_losses)
            assert torch.isfinite(values).all(), "nonfinite training loss"
            profile["losses"] = values.tolist()
        published = optimizer.snapshot()
        validation.check_publication(published, dp_group)
        total_steps = profile["next_step"]
    final_weights = validation.fingerprints(published)
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
                  rank=rank, stage=index, data_rank=parallel.data_rank,
                  stage_local_rank=replica, pp=args.stages, dp=replicas,
                  cp=args.cp, ep=args.ep, packed=True, packing=packs,
                  cp_rank=parallel.cp_rank, ep_rank=parallel.ep_rank,
                  expert_gradient_group_size=dist.get_world_size(parallel.expert_group),
                  routing_check="replay-near-ties", routing_comparisons=local_model.routing_comparisons,
                  local_tokens=local_tokens, data_lanes=parallel.data_size,
                  cp_ep_contracts=list(parallel.contracts.values()),
                  microbatches=args.microbatches, tokens=args.tokens, steps=reports,
                  total_training_steps=total_steps, final_weight_sha256=final_weights,
                  final_weight_samples=validation.weight_samples(published),
                  optimizer="dmuon", activation_checkpointing=True,
                  ns_backend="direct", forward_unshard="broadcast",
                  forward_unshard_scope="attention matrices; packed expert MatrixLayout uses owner broadcast",
                  boundary="layer", overlap=args.overlap, hsdp_replicas=args.replicas, hsdp_shards=shards,
                  forward_prefetch_depth=int(args.overlap == "on"),
                  post_step_prefetch_groups=int(args.overlap == "on" and args.replicas == 1),
                  dmuon_version=sharding.dmuon.__version__,
                  loss="global valid-target mean plus valid-target-weighted per-pack HF router auxiliary loss",
                  schedule="torch.distributed.pipelining.Schedule1F1B", sources=SOURCES,
                  torch=torch.__version__, transformers=transformers.__version__, nccl=torch.cuda.nccl.version(),
                  dtype="float32", attention="fa4",
                  numerical_policy=numerical_policy,
                  attention_dtype=str(attention.DTYPE).removeprefix("torch."),
                  flash_attn=version("flash-attn-4"),
                  physical_hosts=len(hosts),
                  logical_stage_groups=[list(range(i * replicas, (i + 1) * replicas)) for i in range(args.stages)],
                  owned_parameters=owned_parameters,
                  peak_allocated_bytes=validation_peak, profile=profile,
                  reference_offload=True,
                  reference_gradient_scope="stage-owned parameters",
                  memory_scope="peak across reference and training phases; reference offloaded between checks",
                  matched_transfers=matched, execution=stage.execution, transfers=stage.transfers,
                  reference_communication=["updated stage weights broadcast to full-model oracles after every step",
                                           "reference gradients averaged across DP peers; dmuon snapshots compared across peers"],
                  communication_model_scope="PP Exchange and observed CP/EP Collective/Exchange contracts; runtime counts specialized")
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({k: v for k, v in result.items()
                          if k not in {"execution", "transfers", "final_weight_sha256", "final_weight_samples", "cp_ep_contracts",
                                       "routing_comparisons"}}, indent=2))

def execute(command, folder, fingerprints):
    print(f"Running {folder.name}; log: {folder.with_suffix('.log')}", flush=True)
    with folder.with_suffix(".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (folder / "source_sha256.json").write_text(json.dumps(fingerprints, indent=2) + "\n")

def compare(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--width", choices=("full", "tiny"), default="full")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true", help="rebuild the report from completed runs")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    from .report import check_pair, summarize, weight_difference

    here = Path(__file__).resolve().parent
    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(here.glob("*.py"))}
    generator = random.Random(20261112)
    cases = []
    for name in args.cases:
        order = generator.sample(["off", "on"], 2)
        variants = {}
        for mode in order:
            folder = args.output / f"{name}-{mode}"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                       "--module", "examples.qwen3_training.train", "run", "--width", args.width, "--layers", str(args.layers), "--stages", "2",
                       "--tokens", str(args.tokens), "--microbatches", "4", "--steps", "2",
                       "--overlap", mode, "--warmup-steps", "2",
                       "--timing-steps", "4", "--timing-rounds", "3", "--profile-steps", "2", "--output", str(folder)]
            for key, value in CASES[name].items():
                command.extend((f"--{key}", str(value)))
            if not args.summarize_only:
                execute(command, folder, fingerprints)
            variants[mode] = summarize(folder)
        check_pair(variants["off"], variants["on"])
        left = [json.loads((args.output / f"{name}-off" / f"rank{rank}.json").read_text()) for rank in range(8)]
        right = [json.loads((args.output / f"{name}-on" / f"rank{rank}.json").read_text()) for rank in range(8)]
        cases.append(dict(name=name, order=order, final_weights=weight_difference(left, right), variants=variants,
                          source_sha256=json.loads((args.output / f"{name}-{order[0]}" / "source_sha256.json").read_text())))
        result = dict(schema=1, width=args.width, scope="complete Qwen training steps; see recorded shape and dtype",
                      recorded_utc=datetime.now(timezone.utc).isoformat(), cases=cases,
                      timing="three multi-step windows per candidate; max-rank window time; median; profiler excluded",
                      profile="CUDA activities and named operation intervals; NCCL includes readiness and protocol waits",
                      comparison="paired execution policies within each configuration; not a ranking across sharding configurations")
        (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {variants['off']['seconds_per_step']:.3f} -> {variants['on']['seconds_per_step']:.3f} s/step", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "compare"))
    args = parser.parse_args(sys.argv[1:2])
    (run if args.command == "run" else compare)(sys.argv[2:])


if __name__ == "__main__":
    main()
