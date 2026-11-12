"""Run matched Qwen training comparisons through the existing pipeline runner."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from timeline import display_intervals, read_rank

CASES = {
    "cp": dict(cp=2, ep=1, replicas=1),
    "cp_ep": dict(cp=2, ep=2, replicas=1),
    "cp_hsdp": dict(cp=2, ep=1, replicas=2),
}
# Configuration differences between candidates invalidate a paired comparison.
MATCH = ("shape", "pp", "dp", "cp", "ep", "packed", "tokens", "microbatches", "dtype", "optimizer",
         "activation_checkpointing", "boundary", "hsdp_replicas", "ns_backend", "forward_unshard",
         "routing_check", "transport", "total_training_steps", "sources")


def summarize(folder):
    paths = sorted(folder.glob("rank*.json"), key=lambda p: int(p.stem[4:]))
    manifests = [json.loads(path.read_text()) for path in paths]
    if not manifests or any(m["status"] != "passed" or not m.get("profile") for m in manifests):
        raise ValueError(f"{folder}: every rank must finish validation and profiling")
    first = manifests[0]
    world = first["pp"] * first["dp"]
    if [m["rank"] for m in manifests] != list(range(world)):
        raise ValueError("rank manifests are incomplete")
    config = {key: first[key] for key in MATCH}
    for m in manifests:
        if any(m[key] != config[key] for key in MATCH):
            raise ValueError("rank manifests disagree on the workload")
    ranks = [read_rank(folder, rank) for rank in range(world)]
    if len({r["host"] for r in ranks}) != 1:
        raise ValueError("timeline clock alignment across hosts has not been established")
    clock = min(r["base"] for r in ranks)
    step = first["profile"]["first_profiled_step"]
    selected = [[e for e in r["kernels"] if e["step"] == step] for r in ranks]
    origin = min(e["ts"] + (r["base"] - clock) / 1000 for r, events in zip(ranks, selected) for e in events)
    lanes = []
    for rank, events in zip(ranks, selected):
        shift = (rank["base"] - clock) / 1000 - origin
        kinds = {kind: display_intervals([(e["ts"], e["ts"] + e["dur"]) for e in events if e["kind"] == kind], shift)
                 for kind in ("forward", "backward", "optimizer", "other", "nccl")}
        operations = {name: display_intervals([(e["ts"], e["ts"] + e["dur"]) for e in events if e["operation"] == name], shift)
                      for name in sorted({e["operation"] for e in events})}
        lanes.append(dict(rank=rank["metrics"]["rank"], intervals=kinds, operations=operations))
    forward = [e for e in selected[0] if e["phase"] == "forward" and e["microbatch"] == "0" and e["layer"] == "0"]
    detail_origin = min((e["ts"] for e in forward), default=0)
    detail = {name: dict(intervals=display_intervals([(e["ts"], e["ts"]+e["dur"]) for e in forward if e["operation"] == name], -detail_origin),
                        kernels=sorted({e["name"] for e in forward if e["operation"] == name})[:3],
                        torch_ops=sorted({e["operator"] for e in forward if e["operation"] == name and e["operator"]})[:3])
              for name in sorted({e["operation"] for e in forward})}
    profile = first["profile"]
    examples = {}
    for contract in first["cp_ep_contracts"]:
        key = contract["kind"], tuple(contract["participants"]), contract["dtype"], len(contract["input_shape"])
        examples.setdefault(key, contract)
    return dict(config=config, overlap=first["overlap"], profile_steps=profile["profiled_steps"],
                seconds_per_step=profile["unprofiled_seconds_per_step"],
                samples_seconds_per_step=profile["samples_seconds_per_step"],
                rank_window_seconds=profile["rank_window_seconds"],
                warmup_steps=profile["warmup_steps"], steps_per_round=profile["steps_per_round"],
                valid_targets_per_step=profile["valid_targets_per_step"],
                valid_targets_per_second=profile["valid_targets_per_second"],
                peak_allocated_bytes=max(m["profile"]["training_peak_allocated_bytes"] for m in manifests),
                numerical_steps=len(first["steps"]),
                max_gradient_error=max(s["max_gradient_error"] for m in manifests for s in m["steps"]),
                changed_routing_sets=sum(s["changed_sets"] for m in manifests for s in m["routing_comparisons"]),
                matched_transfers=first["matched_transfers"],
                cp_ep_contracts=sorted({c["kind"] for m in manifests for c in m["cp_ep_contracts"]}),
                metrics=[r["metrics"] for r in ranks], first_step=lanes, layer_detail=detail,
                contract_examples=list(examples.values()),
                pp_examples=[e for e in first["transfers"] if e["step"] == 0 and e["microbatch"] == 0],
                software={k: first[k] for k in ("torch", "transformers", "nccl", "dmuon_version")})


def check_pair(left, right):
    if left["config"] != right["config"]:
        raise ValueError("paired candidates must use the same model, data and sharding configuration")
    for key in ("valid_targets_per_step", "warmup_steps", "steps_per_round", "profile_steps", "numerical_steps"):
        if left[key] != right[key]:
            raise ValueError(f"paired candidates disagree on {key}")
    if len(left["samples_seconds_per_step"]) != len(right["samples_seconds_per_step"]):
        raise ValueError("paired candidates need identical timing windows")


def weight_difference(left, right):
    """Report drift without assuming bitwise deterministic training or relaxing the oracle."""
    count, different, maximum, squared, baseline, samples = 0, 0, 0.0, 0.0, 0.0, 0
    for a, b in zip(left, right):
        # Published weights are replicated; count one copy per pipeline stage.
        if a["stage_local_rank"]:
            continue
        for name, fingerprint in a["final_weight_sha256"].items():
            count += 1
            different += fingerprint != b["final_weight_sha256"][name]
            for x, y in zip(a["final_weight_samples"][name], b["final_weight_samples"][name]):
                maximum = max(maximum, abs(x-y))
                squared += (x-y)**2
                baseline += x*x
                samples += 1
    return dict(parameter_tensors=count, different_hashes=different, sampled_elements=samples,
                sample_max_abs=maximum, sample_relative_l2=math.sqrt(squared / baseline) if baseline else None,
                scope="all-tensor hashes and finite checks; fixed samples diagnose drift, not full-weight equivalence")


def execute(command, folder, fingerprints):
    print(f"Running {folder.name}; log: {folder.with_suffix('.log')}", flush=True)
    with folder.with_suffix(".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ | {"OMP_NUM_THREADS": "1"})
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (folder / "source_sha256.json").write_text(json.dumps(fingerprints, indent=2) + "\n")


def repeat_control(root, name):
    left, right = [[json.loads((root / suffix / f"rank{rank}.json").read_text()) for rank in range(8)]
                   for suffix in (f"{name}-off", f"{name}-off-repeat")]
    for a, b in zip(left, right):
        if any(a[key] != b[key] for key in (*MATCH, "overlap")):
            raise ValueError("unchanged-policy control changed the workload")
    return dict(case=name, overlap="off", samples_seconds_per_step=right[0]["profile"]["samples_seconds_per_step"],
                final_weights=weight_difference(left, right),
                max_gradient_error=max(s["max_gradient_error"] for m in right for s in m["steps"]),
                loss_max_abs_delta=max(abs(x-y) for a, b in zip(left, right)
                                       for x, y in zip(a["profile"].get("losses", []), b["profile"].get("losses", []))),
                source_sha256=json.loads((root / f"{name}-off-repeat" / "source_sha256.json").read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--width", choices=("full", "tiny"), default="full")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true", help="rebuild the report from completed runs")
    parser.add_argument("--repeat-baseline", action="store_true", help="repeat the first case's off policy to diagnose run-to-run drift")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    sources = ("pipeline.py", "pipeline_model.py", "pipeline_trace.py", "config.py", "packed.py", "parallel.py",
               "parallel_optimizer.py", "dedicated.py", "profiling.py")
    fingerprints = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in sources}
    generator = random.Random(20261112)
    cases = []
    for name in args.cases:
        order = generator.sample(["off", "on"], 2)
        variants = {}
        for mode in order:
            folder = args.output / f"{name}-{mode}"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                       str(here / "pipeline.py"), "--width", args.width, "--layers", str(args.layers), "--stages", "2",
                       "--tokens", str(args.tokens), "--microbatches", "4", "--steps", "2", "--packed",
                       "--optimizer", "dmuon", "--checkpoint", "stage", "--overlap", mode,
                       "--routing-check", "replay-near-ties", "--reference-offload", "--warmup-steps", "2",
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
                      profile="CUDA kernel unions; NCCL includes waits; first step on a shared host clock, 100 us display grid",
                      comparison="paired execution policies within each configuration; not a ranking across sharding configurations")
        (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {variants['off']['seconds_per_step']:.3f} -> {variants['on']['seconds_per_step']:.3f} s/step", flush=True)
    name = args.cases[0]
    folder = args.output / f"{name}-off-repeat"
    if args.repeat_baseline and not args.summarize_only:
        command = json.loads((args.output / f"{name}-off" / "command.json").read_text())
        command[command.index("--output") + 1] = str(folder)
        execute(command, folder, fingerprints)
    if args.repeat_baseline or (folder / "rank0.json").exists():
        result["unchanged_policy_control"] = repeat_control(args.output, name)
        (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
