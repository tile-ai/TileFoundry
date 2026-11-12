"""Validate and measure the four authored gradient programs; profile only after timing."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import tvm_ffi
from config import FULL
from overlap import MODES, program
from runtime import Communicator
from timeline import duration, intersection

from tilefoundry import analysis, codegen, inspection, runtime
from tilefoundry.ir import expr, tir
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda


def report(mod, plan):
    """Read legality, fusion boundaries and explicit memory from the core proof."""
    host = mod.entry_function()
    checked = analysis.check_communication(host, module=mod)
    overlaps = analysis.overlap_candidates(checked)
    kernels = [op for op in checked.operations if isinstance(op.statement.callable, tir.launch.Launch)]
    local_bytes = sum(effect.region.relation.range().count_val().get_num_si() * effect.mode.value.bit_count()
                      for op in kernels for effect in op.effects)
    boundaries = []
    for op in checked.operations:
        if not isinstance(op.statement.callable, C.CommEnqueue) or op.statement.callable.operation.reduction != "sum":
            continue
        destination = op.statement.args[1].name
        consumer = next(other for other in kernels if (destination, expr.MemoryEffect.READ) in other.accesses)
        boundary = analysis.region_boundary(checked, (op.name, consumer.name))
        boundaries.append(dict(operations=boundary.operations, inputs=len(boundary.inputs), outputs=len(boundary.outputs),
                               prerequisites=boundary.prerequisites, dependents=boundary.dependents))
    return dict(legality="checked", workspace_bytes=plan.peak_bytes, constraint_precision=plan.constraint_precision.value,
                local_kernel_bytes=local_bytes, local_kernel_launches=len(kernels), collective_calls=len(checked.operations) - len(kernels),
                overlap_candidates=overlaps, fusion_boundaries=boundaries,
                workspace_offsets={name: region.offset for name, region in plan.bindings.regions})


def input_pair(rank, step, elements):
    generator = torch.Generator().manual_seed(173 + 1009 * step + 37 * rank)
    return torch.randn(2, elements, generator=generator)


def reference(world, step, elements):
    """FP64 oracle from independently generated rank inputs, without a collective."""
    result = torch.zeros(elements, dtype=torch.float64)
    for rank in range(world):
        values = input_pair(rank, step, elements)
        result += values[0].double() + values[1].double()
    return result / world


class Native:
    """Own the native resources and checked arena for one selected host entry."""

    def __init__(self, mod, plan, communicator, rank, chunks, elements, inputs, outputs):
        self.compute, self.transfer = torch.cuda.Stream(), torch.cuda.Stream()
        self.arena = torch.empty(plan.peak_bytes, dtype=torch.uint8, device="cuda")
        self.events = [torch.cuda.Event() for _ in range(2 * chunks)]
        values = dict(compute=self.compute.cuda_stream, transfer=self.transfer.cuda_stream, comm=communicator.value.value)
        for i in range(chunks):
            values.update({f"a{i}": inputs[0, i], f"b{i}": inputs[1, i], f"y{i}": outputs[i]})
            for offset, name in ((i, f"ready{i}"), (chunks + i, f"done{i}")):
                self.events[offset].record()
                values[name] = self.events[offset].cuda_event
        for name, region in plan.bindings.regions:
            values[name] = self.arena[region.offset:region.offset + elements * 4].view(torch.float32)
        self.fn = tvm_ffi.get_global_func(f"{mod.name}.{mod.entry}")
        self.arguments = (rank, *(values[param.name] for param in mod.entry_function().params))
        self.compute.wait_stream(torch.cuda.current_stream())

    def __call__(self):
        self.fn(*self.arguments)


def profile_summary(path, modes, steps):
    """Exact device intervals inside synchronized host steps; no clock merging."""
    import gzip

    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    windows = [event for event in events if event.get("cat") == "user_annotation" and event["name"].startswith("bucket/")]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    result = {}
    for mode in modes:
        intervals, timeline, counts, collective_kernels = {"compute": [], "nccl": []}, [], Counter(), Counter()
        selected = [window for window in windows if window["name"].split("/")[1] == mode]
        if len(selected) != steps:
            raise ValueError("profile must contain every synchronized bucket step")
        for index, window in enumerate(selected):
            held = [event for event in kernels if window["ts"] <= event["ts"] and event["ts"] + event["dur"] <= window["ts"] + window["dur"]]
            if not held or not any("nccl" in event["name"].lower() for event in held):
                raise ValueError("CUDA profiler did not capture the generated kernels and NCCL")
            origin = min(event["ts"] for event in held)
            for event in held:
                kind = "nccl" if "nccl" in event["name"].lower() else "compute"
                intervals[kind].append((event["ts"], event["ts"] + event["dur"]))
                phase = "collective" if kind == "nccl" else "normalize" if "normalize" in event["name"] else "accumulate"
                counts[phase] += 1
                if kind == "nccl":
                    collective_kernels[event["name"]] += 1
                if index == 0:
                    timeline.append(dict(phase=phase, start_us=event["ts"] - origin, duration_us=event["dur"]))
        communication = duration(intervals["nccl"])
        overlap = intersection(intervals["compute"], intervals["nccl"])
        result[mode] = dict(compute_us=duration(intervals["compute"]) / steps, nccl_us=communication / steps,
                            overlap_us=overlap / steps, nccl_overlap_fraction=overlap / communication,
                            kernels_per_step={phase: count / steps for phase, count in counts.items()},
                            collective_kernels=dict(collective_kernels),
                            first_step=sorted(timeline, key=lambda event: event["start_us"]))
    return result


def benchmark_case(target, args, total, rank, world, communicator):
    chunks, elements = args.chunks, total // args.chunks
    folder = args.output / str(total)
    folder.mkdir(parents=True, exist_ok=True)
    programs = {mode: program(target, world, chunks, elements, mode=mode) for mode in MODES}
    facts = {mode: report(*programs[mode]) for mode in MODES}
    first = programs[MODES[0]][0]
    mod = replace(first, functions=(*first.functions[:2], *(programs[mode][0].entry_function() for mode in MODES)))
    built = [None]
    started = time.perf_counter()
    if rank == 0:
        (folder / "program.py").write_text(inspection.as_script(mod))
        built[0] = codegen.link_program(mod, workdir=folder / "build", options=codegen.Options(communication=cuda.codegen.NCCL))
    dist.broadcast_object_list(built, src=0)
    compile_seconds = time.perf_counter() - started
    runtime.load_linked_module(built[0])
    inputs = torch.empty(2, chunks, elements, device="cuda")
    outputs = torch.empty(chunks, elements, device="cuda")
    natives = {mode: Native(*programs[mode], communicator, rank, chunks, elements, inputs, outputs) for mode in MODES}
    error = {mode: 0.0 for mode in MODES}
    for step in range(args.steps):
        inputs.copy_(input_pair(rank, step, total).reshape(2, chunks, elements))
        expected = reference(world, step, total).reshape(chunks, elements)
        torch.cuda.synchronize()
        for mode in MODES:
            natives[mode]()
            actual = outputs.cpu().double()
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-6)
            error[mode] = max(error[mode], (actual - expected).abs().max().item())
    for mode in MODES:
        for _ in range(args.warmup):
            natives[mode]()
    samples = {mode: [] for mode in MODES}
    order = list(MODES)
    generator = random.Random(20261112)
    orders = []
    for _ in range(args.rounds):
        generator.shuffle(order)
        orders.append(list(order))
        for mode in order:
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                natives[mode]()
            elapsed = (time.perf_counter() - started) * 1e6 / args.iterations
            times = [None] * world
            dist.all_gather_object(times, elapsed)
            samples[mode].append(times)
    errors = [None] * world
    dist.all_gather_object(errors, error)
    results = {}
    for mode in MODES:
        values = [max(sample) for sample in samples[mode]]
        results[mode] = facts[mode] | dict(numerical_check="passed", max_abs_error=max(data[mode] for data in errors),
                                          samples_us=values, rank_samples_us=samples[mode], median_us=statistics.median(values))
    return dict(elements=total, payload_bytes=total * 4, input_output_bytes=total * 12, chunk_elements=elements, chunks=chunks,
                compile_and_distribution_seconds=compile_seconds, timing_order=orders, variants=results), natives


def capture_case(case, natives, args, rank, world):
    """Capture CUDA traces only after every unprofiled measurement has finished."""
    trace = args.output / str(case["elements"]) / f"profile-rank{rank}.json.gz"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for mode in MODES:
            natives[mode]()
            dist.barrier()
            for step in range(args.profile_steps):
                with torch.profiler.record_function(f"bucket/{mode}/{step}"):
                    natives[mode]()
    profiler.export_chrome_trace(str(trace))
    profiled = profile_summary(trace, MODES, args.profile_steps)
    for mode, profile in profiled.items():
        counts = profile["kernels_per_step"]
        chunks = case["chunks"]
        if counts.get("accumulate", 0) != chunks or counts.get("normalize", 0) != (0 if mode.startswith("fused") else chunks):
            raise ValueError("profiled compute kernels do not match the authored realization")
    ranks = [None] * world
    dist.all_gather_object(ranks, profiled)
    for mode in MODES:
        case["variants"][mode]["profiles"] = [data[mode] for data in ranks]


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("overlap.py"), root / "src/tilefoundry/analysis/kernel_effects.py", root / "src/tilefoundry/analysis/execution.py",
             root / "src/tilefoundry/analysis/communication.py", root / "targets/nvidia/tilefoundry_nvidia/cuda/codegen/communication.py",
             root / "targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/nccl.h"]
    return dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                architecture="sm_" + "".join(map(str, torch.cuda.get_device_capability())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, nargs="+", default=[65536, FULL.hidden * FULL.expert_hidden, FULL.hidden * FULL.q_heads * FULL.head_dim],
                        help="total FP32 elements per rank and bucket")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5, help="fresh-input correctness steps")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--profile-steps", type=int, default=3)
    args = parser.parse_args()
    if min(args.chunks, args.steps, args.warmup, args.iterations, args.profile_steps) < 1 or args.rounds < 3:
        parser.error("require positive counts and at least three timing rounds")
    if any(n < 1024 * args.chunks or n % (1024 * args.chunks) for n in args.elements):
        parser.error("bucket sizes must be positive multiples of chunks * 1024")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    if args.output.exists():
        raise FileExistsError("choose a new evidence directory")
    dist.barrier()
    args.output.mkdir(parents=True, exist_ok=True)
    base = cuda.Target("nvidia.h200_sxm", device_count=world)
    major, minor = torch.cuda.get_device_capability()
    architecture = replace(base.architecture, name=f"sm_{major}{minor}", gpu_architecture=f"compute_{major}{minor}", gpu_code=(f"sm_{major}{minor}",))
    target = cuda.Target(base.device, architecture, device_count=world)
    communicator = Communicator(tuple(range(world)), rank)
    measured = [benchmark_case(target, args, total, rank, world, communicator) for total in args.elements]
    for case, natives in measured:
        capture_case(case, natives, args, rank, world)
    cases = [case for case, _ in measured]
    communicator.close()
    if rank == 0:
        result = dict(schema=1, status="passed", recorded_utc=datetime.now(timezone.utc).isoformat(),
                      world=world, steps=args.steps, warmup=args.warmup, iterations=args.iterations,
                      rounds=args.rounds, profile_steps=args.profile_steps, provenance=provenance(), cases=cases,
                      correctness=dict(oracle="independent FP64 CPU sum of seeded per-rank inputs", rtol=3e-6, atol=2e-6),
                      semantics="FP32 accumulation of two microbatch gradients, then mean across ranks; synthetic inputs, not a complete optimizer step",
                      timing="all latency measurements precede profiler captures; synchronized native host calls; max rank time per batch; randomized variant order; allocation/compilation/input generation excluded",
                      memory="core-planned caller workspace; excludes inputs, outputs and internal NCCL allocations",
                      profile_scope="separate CUPTI kernel capture; rank-local clocks; NCCL includes waits, overlap is not wire time")
        (args.output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({str(case["elements"]): {mode: data["median_us"] for mode, data in case["variants"].items()} for case in cases}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
