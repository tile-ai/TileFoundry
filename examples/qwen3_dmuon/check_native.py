"""Check generated NCCL calls, runtime counts and stream protocols against HIR."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import tvm_ffi

from tilefoundry import codegen, core, runtime
from tilefoundry.interfaces import evaluator, typeinfer
from tilefoundry.ir import core as ir
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C
from tilefoundry_nvidia import cuda

from check_primitives import cases
from model import collective_function


def variable(name, shape, dtype=expr.DType.f32, storage=expr.StorageKind.GMEM):
    return expr.Var(name=name, type=types.TensorType.from_shape(shape, dtype=dtype, storage=storage))


def handle(name, kind, mesh=None, context=None):
    return expr.Var(name=name, type=types.CommunicationHandleType(kind, context or name, mesh))


def effect(op, *args):
    return ir.Evaluate(op, args)


def primitive(name, operation, shape, dtype):
    semantic = collective_function(operation, shape, dtype)
    source = expr.Var(name="source", type=semantic.params[0].type)
    target = expr.Var(name="destination", type=semantic.return_type)
    comm = handle("comm", "communicator", operation.participants, operation.context)
    stream = handle("stream", "stream")
    params = (source, target, comm, stream)
    return ir.PrimFunction(name, params, ir.Sequential((
        effect(C.CommEnqueue(operation=operation), *params),
        effect(C.StreamSynchronize(), stream),
    )), output_count=0)


def dynamic_protocol(mesh):
    n = len(types.Placement.from_mesh(mesh))
    counts_row = variable("count_row", (n,), expr.DType.i64)
    matrix = variable("matrix", (n * n,), expr.DType.i64)
    source, target, reduced = (variable(name, (3 * n,)) for name in ("source", "destination", "reduced"))
    comm = handle("comm", "communicator", mesh, "runtime")
    metadata_stream, payload_stream = (handle(name, "stream") for name in ("metadata_stream", "payload_stream"))
    ready, done = (handle(name, "event") for name in ("ready", "done"))
    gather = sharding.Collective(participants=mesh, context="runtime", kind="all_gather")
    exchange = sharding.Exchange(participants=mesh, context="runtime", receive_capacity=3 * n)
    reduce = sharding.Collective(participants=mesh, context="runtime", kind="all_reduce", reduction="sum")
    params = (counts_row, matrix, source, target, reduced, comm, metadata_stream, payload_stream, ready, done)
    statements = (
        effect(C.CommEnqueue(operation=gather), counts_row, matrix, comm, metadata_stream),
        effect(C.EventRecord(), ready, metadata_stream),
        effect(C.StreamWaitEvent(), payload_stream, ready),
        effect(C.CommEnqueue(operation=exchange), source, target, comm, payload_stream, matrix),
        effect(C.CommEnqueue(operation=reduce), target, reduced, comm, payload_stream),
        effect(C.EventRecord(), done, payload_stream),
        effect(C.ExportEvent(), done),
    )
    return ir.PrimFunction("dynamic", params, ir.Sequential(statements), output_count=0)


def grouped_protocol(mesh):
    source, prepared = (variable(name, (4, 8)) for name in ("source", "prepared"))
    a, b, total = (variable(name, (4, 1)) for name in ("a", "b", "total"))
    comm = handle("comm", "communicator", mesh, "grouped")
    other = handle("other", "communicator", mesh, "independent")
    s0, s1 = (handle(name, "stream") for name in ("s0", "s1"))
    op = sharding.Collective(participants=mesh, context="grouped", kind="all_reduce", reduction="sum")
    scatter = sharding.Collective(participants=mesh, context="grouped", kind="reduce_scatter", reduction="sum", axis=-1)
    independent = sharding.Collective(participants=mesh, context="independent", kind="reduce_scatter", reduction="sum", axis=-1)
    params = (source, prepared, a, b, total, comm, other, s0, s1)
    body = ir.Sequential((
        effect(C.CommEnqueue(operation=op), source, prepared, comm, s0),
        effect(C.CommGroupStart()),
        effect(C.CommEnqueue(operation=scatter), prepared, a, comm, s0),
        effect(C.CommGroupStart()),
        effect(C.CommEnqueue(operation=independent), prepared, b, other, s1),
        effect(C.CommGroupEnd()), effect(C.CommGroupEnd()),
        effect(C.CommEnqueue(operation=op), b, total, comm, s0),
        effect(C.StreamSynchronize(), s0),
    ))
    return ir.PrimFunction("grouped", params, body, output_count=0)


def counted_exchange(mesh, storage, dtype):
    name = f"counted_{storage.value}_{dtype.name}"
    n = len(types.Placement.from_mesh(mesh))
    src, dst = (variable(name, (3 * n,)) for name in ("source", "destination"))
    counts = variable("counts", (n, n), dtype, storage)
    comm = handle("comm", "communicator", mesh, "counts")
    stream = handle("stream", "stream")
    op = sharding.Exchange(participants=mesh, context="counts", receive_capacity=3 * n)
    params = (src, dst, comm, stream, counts)
    body = ir.Sequential((effect(C.CommEnqueue(operation=op), *params), effect(C.StreamSynchronize(), stream)))
    return ir.PrimFunction(name, params, body, output_count=0)


class Communicator:
    """Bootstrap only; all data-plane operations come from TileFoundry's binary."""

    def __init__(self, ranks, rank):
        self.lib = ctypes.CDLL(str(Path(os.environ["NCCL_HOME"]) / "lib/libnccl.so.2"))
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        class unique_type(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_byte * 128)]
        unique = unique_type()
        if rank == 0:
            self.check(self.lib.ncclGetUniqueId(ctypes.byref(unique)))
        data = [bytes(unique)]
        dist.broadcast_object_list(data, src=0)
        unique = unique_type.from_buffer_copy(data[0])
        self.value = ctypes.c_void_p()
        if rank in ranks:
            self.lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_type, ctypes.c_int]
            self.check(self.lib.ncclCommInitRank(ctypes.byref(self.value), len(ranks), unique, ranks.index(rank)))
        dist.barrier()

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.ncclGetErrorString(status).decode())

    def close(self):
        if self.value:
            self.check(self.lib.ncclCommDestroy(self.value))


def oracle_exchange(mesh, counts, sources, capacity):
    source = variable("source", (capacity,))
    metadata = variable("counts", (len(counts), len(counts)), expr.DType.i64, expr.StorageKind.HOST)
    op = sharding.Exchange(participants=mesh, context="oracle", receive_capacity=capacity)
    call = expr.Call(op, (source, metadata), type=source.type)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    function = ir.Function.from_parts(name="exchange", params=(source, metadata), body=call, return_type=call.type)
    return evaluator.evaluate(function, core.ShardTensor(source.type, sources),
                              core.ShardTensor(metadata.type, torch.tensor(counts, dtype=torch.int64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 8:
        parser.error("use torchrun --nproc-per-node=8")
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / f"rank{rank}.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    os.environ.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,ENV",
                      NCCL_DEBUG_FILE=str(args.output.resolve() / f"nccl-rank{rank}.log"))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
    members = types.Placement.from_mesh(mesh)
    tests = cases(mesh)
    functions = [primitive(f"case{i}", *case) for i, case in enumerate(tests)]
    functions += [dynamic_protocol(mesh), grouped_protocol(mesh)]
    functions += [counted_exchange(mesh, storage, dtype) for storage in (expr.StorageKind.HOST, expr.StorageKind.GMEM)
                  for dtype in (expr.DType.i32, expr.DType.i64)]
    module = ir.Module(name="NativeChecks", functions=tuple(functions), entry=functions[0].name,
                       target=cuda.Target("nvidia.h200_sxm", device_count=8), topologies=mesh.topologies)
    if rank == 0:
        linked = codegen.link_program(module, workdir=args.output / "build",
                                      options=codegen.Options(communication=cuda.codegen.NCCL))
        (args.output / "library.txt").write_text(str(linked.library_path))
    dist.barrier()
    loaded = tvm_ffi.load_module((args.output / "library.txt").read_text())
    runtime.loader.libraries.keep(loaded)
    def call(name, *values):
        return tvm_ffi.get_global_func(f"NativeChecks.{name}")(rank, *values)
    groups = {}
    for op, _, _ in tests:
        ranks = tuple(p.ids["gpu"] for p in types.Placement.from_mesh(op.participants))
        if ranks not in groups:
            groups[ranks] = Communicator(ranks, rank)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    placement = types.Placement({"gpu": rank})
    for i, (op, shape, dtype) in enumerate(tests):
        places = types.Placement.from_mesh(op.participants)
        ranks = tuple(p.ids["gpu"] for p in places)
        if rank not in ranks:
            continue
        pieces = tuple((p, torch.arange(math.prod(shape)).reshape(shape).to(dtype) + p.ids["gpu"]) for p in places)
        function = collective_function(op, shape, dtype)
        expected = evaluator.evaluate(function, core.ShardTensor(function.params[0].type, pieces))
        source = next(value for p, value in pieces if p == placement).cuda()
        destination = torch.empty(function.return_type.shape, dtype=dtype, device="cuda")
        streams[0].wait_stream(torch.cuda.current_stream())
        call(f"case{i}", source, destination, groups[ranks].value.value, streams[0].cuda_stream)
        if placement in expected.placements:
            torch.testing.assert_close(destination.cpu(), expected.to_local(placement), rtol=0, atol=0)
    ranks = tuple(p.ids["gpu"] for p in members)
    comm = groups[ranks].value.value
    independent = Communicator(ranks, rank)
    capacity = 3 * world
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record(streams[0])
    done.record(streams[1])
    for step in range(16):
        counts = tuple(tuple((i * j + step) % 4 if (i + step) % 3 else 0 for j in range(world)) for i in range(world))
        pieces = tuple((p, torch.arange(capacity).float() + p.ids["gpu"] * 100) for p in members)
        expected = oracle_exchange(mesh, counts, pieces, capacity)
        total = torch.stack([expected.to_local(p) for p in members]).sum(0)
        row = torch.tensor(counts[ranks.index(rank)], dtype=torch.int64, device="cuda")
        matrix = torch.empty(world * world, dtype=torch.int64, device="cuda")
        source = next(value for p, value in pieces if p == placement).cuda()
        destination, reduced = (torch.empty_like(source) for _ in range(2))
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        call("dynamic", row, matrix, source, destination, reduced, comm, *(s.cuda_stream for s in streams), ready.cuda_event, done.cuda_event)
        done.synchronize()
        torch.testing.assert_close(matrix.cpu().reshape(world, world), torch.tensor(counts), rtol=0, atol=0)
        torch.testing.assert_close(destination.cpu(), expected.to_local(placement), rtol=0, atol=0)
        torch.testing.assert_close(reduced.cpu(), total, rtol=0, atol=0)
    count_modes = []
    for storage in (expr.StorageKind.HOST, expr.StorageKind.GMEM):
        for dtype, torch_dtype in ((expr.DType.i32, torch.int32), (expr.DType.i64, torch.int64)):
            counts_tensor = torch.tensor(counts, dtype=torch_dtype,
                                         device="cpu" if storage == expr.StorageKind.HOST else "cuda")
            name = counted_exchange(mesh, storage, dtype).name
            streams[0].wait_stream(torch.cuda.current_stream())
            call(name, source, destination, comm, streams[0].cuda_stream, counts_tensor)
            torch.testing.assert_close(destination.cpu(), expected.to_local(placement), rtol=0, atol=0)
            count_modes.append(name)
    rejected = []
    host_counts = counted_exchange(mesh, expr.StorageKind.HOST, expr.DType.i64).name
    for reason, bad in (("negative", torch.full((world, world), -1, dtype=torch.int64)),
                        ("capacity", torch.full((world, world), capacity, dtype=torch.int64))):
        try:
            call(host_counts, source, destination, comm, streams[0].cuda_stream, bad)
        except RuntimeError as error:
            assert "negative or exceeds capacity" in str(error)
            rejected.append(reason)
        else:
            raise AssertionError(f"invalid {reason} counts were accepted")
    try:
        call(host_counts, source, source, comm, streams[0].cuda_stream, torch.tensor(counts))
    except RuntimeError as error:
        assert "storage alias" in str(error)
        rejected.append("alias")
    else:
        raise AssertionError("undeclared buffer alias was accepted")
    source = torch.arange(32., device="cuda").reshape(4, 8) + rank
    prepared = torch.empty_like(source)
    a, b, total = (torch.empty((4, 1), device="cuda") for _ in range(3))
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    call("grouped", source, prepared, a, b, total, comm, independent.value.value, *(s.cuda_stream for s in streams))
    whole = torch.arange(32.).reshape(4, 8).mul(world).add(sum(ranks)).mul(world)
    index = ranks.index(rank)
    expected = whole[:, index:index + 1]
    torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(total.cpu(), whole.sum(dim=1, keepdim=True), rtol=0, atol=0)
    dist.barrier()
    for group in groups.values():
        group.close()
    independent.close()
    result_file.write_text(json.dumps(dict(status="passed", rank=rank, primitive_cases=len(tests),
                                         dynamic_steps=16, nested_multistream_group=True, independent_contexts=True,
                                         runtime_count_modes=count_modes, rejected=rejected,
                                         backend="generated NCCL/CUDA", nccl=torch.cuda.nccl.version(),
                                         transport=os.environ.get("NCCL_NET", "native")), indent=2) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        print(f"Passed {len(tests)} generated primitive cases, 16 dynamic protocols and nested groups on eight GPUs")


if __name__ == "__main__":
    main()
