"""Compare serialized and chunk-pipelined compute/collective regions from one TIR program."""

import argparse
from dataclasses import replace
from itertools import combinations
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import tvm_ffi

from tilefoundry import analysis, codegen, runtime
from tilefoundry.dsl import B, Layout, Mesh, ShardLayout, T, Tensor, Topology, module, prim_func
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda

from check_native import Communicator, handle, variable
from simulate import Graph


def compute_kernel(target, elements):
    @module(target=target)
    class Compute:
        @prim_func(target=target)
        def twice(x: Tensor[(elements,), "f32"], y: Tensor[(elements,), "f32"]):
            with Mesh((Topology("thread", 1),), Layout((1,))) as thread:
                fragment = T.alloc_tensor(tensor_type=Tensor[
                    (elements,), "f32", ShardLayout(Layout((elements,)), (B(),), thread), "rmem"])
                source = T.tensor_view(T.ptr_of(x), layout=ShardLayout(Layout((elements,)), (B(),), thread))
                destination = T.tensor_view(T.ptr_of(y), layout=ShardLayout(Layout((elements,)), (B(),), thread))
                T.copy(source, fragment)
                T.binary(fragment, fragment, fragment, kind=expr.BinaryKind.ADD)
                T.copy(fragment, destination)
    return Compute.functions[0]


def program(target, world=2, chunks=2, elements=32, *, pipelined=False, capacity_bytes=None):
    if world < 1 or chunks < 1 or elements < 1:
        raise ValueError("world, chunks and elements must be positive")
    kernel = compute_kernel(target, elements)
    mesh = types.Mesh((types.Topology("gpu", world),), types.Layout((world,)), ("rank",))
    tensors = {f"{name}{i}": variable(f"{name}{i}", (elements,))
               for i in range(chunks) for name in ("x", "prepared", "received", "y")}
    compute, transfer = handle("compute", "stream"), handle("transfer", "stream")
    comm = handle("comm", "communicator", mesh, "region")
    ready = [handle(f"ready{i}", "event") for i in range(chunks)]
    done = [handle(f"done{i}", "event") for i in range(chunks)]
    collective = sharding.Collective(participants=mesh, context="region", kind="all_reduce", reduction="sum")
    statements, consumers = [], []
    for i in range(chunks):
        statements += [
            tir.launch.launch_call(kernel, (tensors[f"x{i}"], tensors[f"prepared{i}"]),
                                   (1, 1, 1), (1, 1, 1), stream=compute),
            core.Evaluate(C.EventRecord(), (ready[i], compute)),
            core.Evaluate(C.StreamWaitEvent(), (transfer, ready[i])),
            core.Evaluate(C.CommEnqueue(operation=collective),
                          (tensors[f"prepared{i}"], tensors[f"received{i}"], comm, transfer)),
            core.Evaluate(C.EventRecord(), (done[i], transfer)),
        ]
        consumer = [core.Evaluate(C.StreamWaitEvent(), (compute, done[i])),
                    tir.launch.launch_call(kernel, (tensors[f"received{i}"], tensors[f"y{i}"]),
                                           (1, 1, 1), (1, 1, 1), stream=compute)]
        (consumers if pipelined else statements).extend(consumer)
    statements += consumers + [core.Evaluate(C.StreamSynchronize(), (compute,))]
    host = core.PrimFunction("host", (*tensors.values(), compute, transfer, comm, *ready, *done),
                            core.Sequential(tuple(statements)), output_count=0)
    name = "Overlap_pipelined" if pipelined else "Overlap_serial"
    mod = core.Module(name=name, functions=(kernel, host), entry="host", target=target, topologies=mesh.topologies)
    scratch = tuple(name for name in tensors if name.startswith(("prepared", "received")))
    plan = analysis.plan_workspace(host, scratch, module=mod, capacity_bytes=capacity_bytes)
    selected = replace(host, metadata=(plan.bindings,))
    mod.replace_functions((kernel, selected))
    return mod, plan


def report(mod, plan, *, kernel_us=None, communication_us=None):
    checked = analysis.check_communication(mod.entry_function(), module=mod)
    overlaps = [(a.name, b.name) for a, b in combinations(checked.operations, 2)
                if a.complete not in checked.predecessors[b.read] and b.complete not in checked.predecessors[a.read]]
    result = dict(legality="checked" if not checked.hazards else "unsafe", workspace_bytes=plan.peak_bytes,
                  operations=len(checked.operations), overlap_candidates=overlaps,
                  workspace_offsets={name: region.offset for name, region in plan.bindings.regions})
    if kernel_us is not None and communication_us is not None:
        graph = Graph()
        for operation in checked.operations:
            kind = "communication" if isinstance(operation.statement.callable, C.CommEnqueue) else "kernel"
            duration = communication_us if kind == "communication" else kernel_us
            deps = [other.name for other in checked.operations if other.complete in checked.predecessors[operation.read]]
            graph.add(operation.name, duration / 1e6, deps, (operation.stream,), name=operation.name,
                      stage=0, lane=operation.stream, step=0)
        seconds, trace = graph.run()
        result.update(estimated_us=seconds * 1e6, cost_provenance="user-supplied, uncontended estimate", trace=trace)
    return result


def run_native(mod, plan, folder, world, rank, chunks, elements, steps):
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        linked = codegen.link_program(mod, workdir=folder / "build", options=codegen.Options(communication=cuda.codegen.NCCL))
        (folder / "library.txt").write_text(str(linked.library_path))
    dist.barrier()
    library = tvm_ffi.load_module((folder / "library.txt").read_text())
    runtime.loader.libraries.keep(library)
    communicator = Communicator(tuple(range(world)), rank)
    compute, transfer = torch.cuda.Stream(), torch.cuda.Stream()
    values = dict(compute=compute.cuda_stream, transfer=transfer.cuda_stream, comm=communicator.value.value)
    arena = torch.empty(plan.peak_bytes, dtype=torch.uint8, device="cuda")
    for name, region in plan.bindings.regions:
        values[name] = arena[region.offset:region.offset + elements * 4].view(torch.float32)
    events = []
    for i in range(chunks):
        values[f"x{i}"] = torch.arange(elements, dtype=torch.float32, device="cuda") + rank + i
        values[f"y{i}"] = torch.empty(elements, device="cuda")
        for prefix in ("ready", "done"):
            event = torch.cuda.Event()
            event.record()
            events.append(event)
            values[f"{prefix}{i}"] = event.cuda_event
    fn = tvm_ffi.get_global_func(f"{mod.name}.host")
    for step in range(steps):
        for i in range(chunks):
            values[f"x{i}"].copy_(torch.arange(elements, dtype=torch.float32, device="cuda") + rank + i + step)
        compute.wait_stream(torch.cuda.current_stream())
        fn(rank, *(values[param.name] for param in mod.entry_function().params))
        for i in range(chunks):
            expected = 4 * (world * (torch.arange(elements, dtype=torch.float32) + i + step) + world * (world - 1) / 2)
            torch.testing.assert_close(values[f"y{i}"].cpu(), expected, rtol=0, atol=0)
    wrong = dict(values)
    wrong[plan.bindings.regions[0][0]] = torch.empty(elements, device="cuda")
    try:
        fn(rank, *(wrong[param.name] for param in mod.entry_function().params))
    except RuntimeError as error:
        if "allocation binding mismatch" not in str(error):
            raise
    else:
        raise AssertionError("native emission accepted an incorrect workspace binding")
    communicator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--elements", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--kernel-us", type=float)
    parser.add_argument("--communication-us", type=float)
    parser.add_argument("--run-native", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if (args.kernel_us is None) != (args.communication_us is None):
        parser.error("provide both kernel-us and communication-us, or neither")
    if args.kernel_us is not None and any(not math.isfinite(value) or value <= 0
                                          for value in (args.kernel_us, args.communication_us)):
        parser.error("cost estimates must be finite and positive")
    rank, world = 0, args.world
    target = cuda.Target("nvidia.h200_sxm", device_count=world)
    if args.run_native:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        major, minor = torch.cuda.get_device_capability()
        # This scalar kernel uses no device rates or architecture-specific instructions.
        # Compile for the executing device without inventing a hardware performance model.
        architecture = replace(target.architecture, name=f"sm_{major}{minor}",
                               gpu_architecture=f"compute_{major}{minor}", gpu_code=(f"sm_{major}{minor}",))
        target = cuda.Target(target.device, architecture, device_count=world)
        dist.init_process_group("gloo")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"rank{rank}.json"
    if destination.exists():
        raise FileExistsError(destination)
    results = {}
    for mode in ("serial", "pipelined"):
        try:
            mod, plan = program(target, world, args.chunks, args.elements, pipelined=mode == "pipelined",
                                capacity_bytes=args.capacity_bytes)
        except analysis.Error as error:
            results[mode] = dict(legality="not selected", reason=str(error))
            continue
        results[mode] = report(mod, plan, kernel_us=args.kernel_us, communication_us=args.communication_us)
        if args.run_native:
            run_native(mod, plan, args.output / mode, world, rank, args.chunks, args.elements, args.steps)
            results[mode]["numerical_check"] = "passed"
            results[mode]["steps"] = args.steps
            results[mode]["incorrect_workspace_rejected"] = True
    destination.write_text(json.dumps(results, indent=2) + "\n")
    if args.run_native:
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
