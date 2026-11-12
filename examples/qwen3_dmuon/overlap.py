"""Four realizations of one gradient boundary: mean_rank(microbatch_a + microbatch_b)."""

from dataclasses import replace

from runtime import handle, variable

from tilefoundry import analysis, verifier
from tilefoundry.dsl import B, Layout, Mesh, S, ShardLayout, T, Tensor, Topology, module, prim_func
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C

MODES = ("serial", "pipelined", "fused", "fused_pipelined")


def compute_kernels(target, elements, world):
    """Four contiguous waves of coalesced elements per 256-thread CTA."""
    ctas = elements // 1024
    factor = 1.0 / world

    @module(target=target)
    class Compute:
        @prim_func(target=target)
        def accumulate(a: Tensor[(elements,), "f32"], b: Tensor[(elements,), "f32"], y: Tensor[(elements,), "f32"]):
            with Mesh((Topology("cta", ctas), Topology("thread", 256)), Layout((ctas, 256))) as workers:
                left = T.alloc_tensor(tensor_type=Tensor[(elements,), "f32", ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers), "rmem"])
                right = T.alloc_tensor(tensor_type=Tensor[(elements,), "f32", ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers), "rmem"])
                first = T.tensor_view(T.ptr_of(a), shape=(elements,), layout=ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers))
                second = T.tensor_view(T.ptr_of(b), shape=(elements,), layout=ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers))
                output = T.tensor_view(T.ptr_of(y), shape=(elements,), layout=ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers))
                T.copy(first, left)
                T.copy(second, right)
                T.binary(left, right, left, kind=expr.BinaryKind.ADD)
                T.copy(left, output)

        @prim_func(target=target)
        def normalize(x: Tensor[(elements,), "f32"], y: Tensor[(elements,), "f32"]):
            with Mesh((Topology("cta", ctas), Topology("thread", 256)), Layout((ctas, 256))) as workers:
                value = T.alloc_tensor(tensor_type=Tensor[(elements,), "f32", ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers), "rmem"])
                scale = T.alloc_tensor(tensor_type=Tensor[(), "f32", ShardLayout(Layout(()), (B(), B()), workers), "rmem"])
                source = T.tensor_view(T.ptr_of(x), shape=(elements,), layout=ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers))
                output = T.tensor_view(T.ptr_of(y), shape=(elements,), layout=ShardLayout(Layout((ctas, 4, 256)), (S(0), S(2)), workers))
                T.copy(source, value)
                T.fill(scale, factor)
                T.binary(value, scale, value, kind=expr.BinaryKind.MUL)
                T.copy(value, output)
    return Compute.functions


def program(target, world=2, chunks=4, elements=4096, *, mode="serial", capacity_bytes=None):
    """All variants compute mean_rank(a + b), with identical bucket/chunk sizes."""
    if mode not in MODES or world < 1 or chunks < 1 or elements < 1024 or elements % 1024:
        raise ValueError("choose a known mode, positive world/chunks and a chunk size divisible by 1024")
    fused, pipelined = mode.startswith("fused"), "pipelined" in mode
    accumulate, normalize = compute_kernels(target, elements, world)
    mesh = types.Mesh((types.Topology("gpu", world),), types.Layout((world,)), ("rank",))
    inputs = {f"{name}{i}": variable(f"{name}{i}", (elements,)) for i in range(chunks) for name in ("a", "b")}
    scratch = {f"{name}{i}": variable(f"{name}{i}", (elements,)) for i in range(chunks)
               for name in (("prepared",) if fused else ("prepared", "received"))}
    outputs = {f"y{i}": variable(f"y{i}", (elements,)) for i in range(chunks)}
    tensors = inputs | scratch | outputs
    compute, transfer = handle("compute", "stream"), handle("transfer", "stream")
    comm = handle("comm", "communicator", mesh, "gradient")
    ready = [handle(f"ready{i}", "event") for i in range(chunks)]
    done = [handle(f"done{i}", "event") for i in range(chunks)]
    collective = sharding.Collective(participants=mesh, context="gradient", kind="all_reduce", reduction="avg" if fused else "sum")
    statements = []

    def launch(kernel, names):
        return tir.launch.launch_call(kernel, tuple(tensors[name] for name in names), (elements // 1024, 1, 1),
                                      (256, 1, 1), stream=compute)

    def consume(i):
        statements.append(core.Evaluate(C.StreamWaitEvent(), (compute, done[i])))
        if not fused:
            statements.append(launch(normalize, (f"received{i}", f"y{i}")))

    depth = 2 if pipelined else 1
    for i in range(chunks):
        if i >= depth:
            consume(i - depth)
        statements.extend((launch(accumulate, (f"a{i}", f"b{i}", f"prepared{i}")),
                           core.Evaluate(C.EventRecord(), (ready[i], compute)),
                           core.Evaluate(C.StreamWaitEvent(), (transfer, ready[i])),
                           core.Evaluate(C.CommEnqueue(operation=collective),
                                         (tensors[f"prepared{i}"], tensors[f"y{i}" if fused else f"received{i}"], comm, transfer)),
                           core.Evaluate(C.EventRecord(), (done[i], transfer))))
    for i in range(max(0, chunks - depth), chunks):
        consume(i)
    statements.append(core.Evaluate(C.StreamSynchronize(), (compute,)))
    host = core.PrimFunction(mode, (*inputs.values(), *scratch.values(), compute, transfer, comm, *ready, *done, *outputs.values()),
                            core.Sequential(tuple(statements)), output_count=chunks)
    kernels = (accumulate,) if fused else (accumulate, normalize)
    mod = core.Module(name=f"Gradient_{elements}", functions=(*kernels, host), entry=mode,
                      target=target, topologies=mesh.topologies)
    verifier.verify_module(mod)
    plan = analysis.plan_workspace(host, tuple(scratch), module=mod, capacity_bytes=capacity_bytes)
    mod.replace_functions((*kernels, replace(host, metadata=(plan.bindings,))))
    return mod, plan
