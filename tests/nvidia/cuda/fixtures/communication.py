"""Cross-rank PyTorch and generated NCCL conformance for generic communication."""

import argparse
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


def collective_function(op, shape, dtype):
    """Infer the primitive once through the ordinary HIR function interface."""
    type_ = types.TensorType.from_shape(tuple(shape), dtype=evaluator.from_torch_dtype(dtype))
    source = expr.Var("payload", type=type_)
    call = expr.Call(op, (source,), type=type_)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    return ir.Function.from_parts(name="communicate", params=(source,), body=call, return_type=call.type)

class Runtime:
    """Bind isolated contexts once, preserving authored participant order."""

    def __init__(self, operations):
        self.groups = {}
        self.trace = []
        for op in operations:
            ranks = tuple(p.ids["gpu"] for p in types.Placement.from_mesh(op.participants))
            if op.context in self.groups:
                if self.groups[op.context][0] != ranks:
                    raise ValueError("a communicator context cannot change participants")
            else:
                self.groups[op.context] = ranks, dist.new_group(list(ranks), sort_ranks=False)

    def apply(self, op, value, shape, device, *, dtype=torch.float32):
        """Return a valid local output or None when only another root receives it."""
        ranks, group = self.groups[op.context]
        rank = dist.get_rank()
        shape = tuple(collective_function(op, shape, dtype).return_type.shape)
        if rank not in ranks:
            return value, shape
        if value is not None and value.dtype != dtype:
            raise TypeError(f"payload dtype {value.dtype} differs from declared {dtype}")
        local = ranks.index(rank)
        if isinstance(op, sharding.Exchange):
            output = torch.zeros(shape, dtype=dtype, device=device)
            send_offset = receive_offset = 0
            operations = []
            for peer, peer_rank in enumerate(ranks):
                sent, received = op.counts[local][peer], op.counts[peer][local]
                if peer == local:
                    if sent:
                        output[receive_offset:receive_offset + received].copy_(value[send_offset:send_offset + sent])
                else:
                    if sent:
                        operations.append(dist.P2POp(dist.isend, value.narrow(0, send_offset, sent), peer_rank, group))
                    if received:
                        operations.append(dist.P2POp(dist.irecv, output.narrow(0, receive_offset, received), peer_rank, group))
                send_offset += sent
                receive_offset += received
            for request in dist.batch_isend_irecv(operations) if operations else ():
                request.wait()
            self.trace.append(dict(context=op.context, kind="exchange", participants=ranks,
                                   communicator_rank=local, counts=op.counts, valid_count=receive_offset,
                                   dtype=str(dtype)))
            return output, shape
        root = ranks[op.root] if op.root is not None else None
        if op.kind == "reduce":
            output = value.clone()
            dist.reduce(output, dst=root, group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
            output = output if rank == root else None
        elif op.kind == "gather":
            buffers = [torch.empty_like(value) for _ in ranks] if rank == root else None
            dist.gather(value, gather_list=buffers, dst=root, group=group)
            output = torch.cat(buffers, dim=op.axis) if buffers is not None else None
        elif op.kind == "scatter":
            output = torch.empty(shape, dtype=dtype, device=device)
            buffers = [x.contiguous() for x in value.chunk(len(ranks), dim=op.axis)] if rank == root else None
            dist.scatter(output, scatter_list=buffers, src=root, group=group)
        elif op.kind == "broadcast":
            output = value.clone() if rank == root else torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(output, src=root, group=group)
        elif op.kind == "all_reduce":
            output = value.clone()
            dist.all_reduce(output, group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
        elif op.kind == "all_gather":
            buffers = [torch.empty_like(value) for _ in ranks]
            dist.all_gather(buffers, value, group=group)
            output = torch.cat(buffers, dim=op.axis)
        elif op.kind == "reduce_scatter":
            output = torch.empty(shape, dtype=dtype, device=device)
            dist.reduce_scatter(output, [x.contiguous() for x in value.chunk(len(ranks), dim=op.axis)],
                                group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
        else:
            raise ValueError(f"unsupported example realization {op.kind}")
        self.trace.append(dict(context=op.context, kind=op.kind, participants=ranks, root=root,
                               communicator_rank=local, output_valid=output is not None, dtype=str(dtype)))
        return output, shape

def variable(name, shape, dtype=expr.DType.f32, storage=expr.StorageKind.GMEM):
    return expr.Var(name=name, type=types.TensorType.from_shape(shape, dtype=dtype, storage=storage))

def handle(name, kind, mesh=None, context=None):
    return expr.Var(name=name, type=types.CommunicationHandleType(kind, context or name, mesh))

def cases(mesh):
    """Exercise dtype, root, rank order, ragged traffic and empty payload contracts."""
    results = []
    size = len(types.Placement.from_mesh(mesh))
    rooted = {"broadcast", "reduce", "gather", "scatter"}
    reducing = {"reduce", "all_reduce", "reduce_scatter"}
    kinds = ("broadcast", "reduce", "all_reduce", "gather", "scatter", "all_gather", "reduce_scatter")
    patterns = dict(
        all_to_all=tuple((1,) * size for _ in range(size)),
        ragged=tuple(tuple((i * j + i) % 3 if i % 3 else 0 for j in range(size)) for i in range(size)),
        ring=tuple(tuple(i + 1 if j == (i + 1) % size else 0 for j in range(size)) for i in range(size)),
        p2p=tuple(tuple(4 if (i, j) == (0, size - 1) else 0 for j in range(size)) for i in range(size)),
        self=tuple(tuple(3 if (i, j) == (2, 2) else 0 for j in range(size)) for i in range(size)),
        zero=tuple((0,) * size for _ in range(size)),
    )
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64):
        for kind in kinds:
            reductions = (("sum", "avg", "max") if dtype.is_floating_point else ("sum", "max"))
            for reduction in reductions if kind in reducing else (None,):
                op = sharding.Collective(participants=mesh, context="ordered", kind=kind,
                                         root=size - 2 if kind in rooted else None, reduction=reduction)
                results.append((op, (size,), dtype))
        for counts in patterns.values():
            op = sharding.Exchange(participants=mesh, context="ordered", counts=counts)
            results.append((op, (max(map(sum, counts)),), dtype))
        op = sharding.Collective(participants=mesh, context="scalar", kind="all_reduce", reduction="sum")
        results.append((op, (), dtype))
    for kind in kinds:
        op = sharding.Collective(participants=mesh, context="empty", kind=kind,
                                 root=size - 2 if kind in rooted else None,
                                 reduction="sum" if kind in reducing else None)
        results.append((op, (0,), torch.float32))
    for kind in ("gather", "scatter", "all_gather", "reduce_scatter"):
        op = sharding.Collective(participants=mesh, context="axis", kind=kind, axis=-1,
                                 root=size - 2 if kind in rooted else None,
                                 reduction="sum" if kind in reducing else None)
        results.append((op, (2, size), torch.float32))
    op = sharding.Collective(participants=mesh[:, 0], context="subgroup", kind="broadcast", root=1)
    results.append((op, (3,), torch.int64))
    return results

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

def oracle_exchange(mesh, counts, sources, capacity):
    source = variable("source", (capacity,))
    metadata = variable("counts", (len(counts), len(counts)), expr.DType.i64, expr.StorageKind.HOST)
    op = sharding.Exchange(participants=mesh, context="oracle", receive_capacity=capacity)
    call = expr.Call(op, (source, metadata), type=source.type)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    function = ir.Function.from_parts(name="exchange", params=(source, metadata), body=call, return_type=call.type)
    return evaluator.evaluate(function, core.ShardTensor(source.type, sources),
                              core.ShardTensor(metadata.type, torch.tensor(counts, dtype=torch.int64)))


def communicator(library, ranks, rank):
    token = [library.unique_id() if rank == 0 else None]
    dist.broadcast_object_list(token, src=0)
    value = library.create(len(ranks), token[0], ranks.index(rank)) if rank in ranks else None
    dist.barrier()
    return value


def native(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.cuda.set_device(types.Placement.from_env().ids["gpu"])
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        parser.error("use torchrun --nproc-per-node=8")
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / f"rank{rank}.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    os.environ.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,ENV",
                      NCCL_DEBUG_FILE=str(args.output.resolve() / f"nccl-rank{rank}.log"))
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
    members = types.Placement.from_mesh(mesh)
    tests = cases(mesh)
    functions = [primitive(f"case{i}", *case) for i, case in enumerate(tests)]
    functions += [dynamic_protocol(mesh), grouped_protocol(mesh)]
    functions += [counted_exchange(mesh, storage, dtype) for storage in (expr.StorageKind.HOST, expr.StorageKind.GMEM)
                  for dtype in (expr.DType.i32, expr.DType.i64)]
    module = ir.Module(name="NativeChecks", functions=tuple(functions), entry=functions[0].name,
                       target=cuda.Target("nvidia.h200_sxm", device_count=8), topologies=mesh.topologies)
    built = [None]
    if rank == 0:
        built[0] = codegen.link_program(module, workdir=args.output / "build",
                                        options=codegen.Options(communication=cuda.codegen.NCCL))
        (args.output / "library.txt").write_text(str(built[0].library_path))
    dist.broadcast_object_list(built, src=0)
    runtime.load_linked_module(built[0])
    library = cuda.nccl.Library(built[0].library_path)
    def call(name, *values):
        return tvm_ffi.get_global_func(f"NativeChecks.{name}")(rank, *values)
    groups = {}
    for op, _, _ in tests:
        ranks = tuple(p.ids["gpu"] for p in types.Placement.from_mesh(op.participants))
        if ranks not in groups:
            groups[ranks] = communicator(library, ranks, rank)
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
        call(f"case{i}", source, destination, groups[ranks], streams[0].cuda_stream)
        if placement in expected.placements:
            torch.testing.assert_close(destination.cpu(), expected.to_local(placement), rtol=0, atol=0)
    ranks = tuple(p.ids["gpu"] for p in members)
    comm = groups[ranks]
    independent = communicator(library, ranks, rank)
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
    call("grouped", source, prepared, a, b, total, comm, independent, *(s.cuda_stream for s in streams))
    whole = torch.arange(32.).reshape(4, 8).mul(world).add(sum(ranks)).mul(world)
    index = ranks.index(rank)
    expected = whole[:, index:index + 1]
    torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(total.cpu(), whole.sum(dim=1, keepdim=True), rtol=0, atol=0)
    dist.barrier()
    for group in groups.values():
        if group is not None:
            library.destroy(group)
    library.destroy(independent)
    result_file.write_text(json.dumps(dict(status="passed", rank=rank, primitive_cases=len(tests),
                                         dynamic_steps=16, nested_multistream_group=True, independent_contexts=True,
                                         runtime_count_modes=count_modes, rejected=rejected,
                                         backend="generated NCCL/CUDA", nccl=torch.cuda.nccl.version(),
                                         transport=os.environ.get("NCCL_NET", "native")), indent=2) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        print(f"Passed {len(tests)} generated primitive cases, 16 dynamic protocols and nested groups on eight GPUs")

def reference(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    device = torch.device("cuda", types.Placement.from_env().ids["gpu"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        parser.error("launch with torchrun --nproc-per-node=8")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"rank{rank}.json"
    if destination.exists():
        raise FileExistsError("select a fresh output directory")
    os.environ.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,P2P,ENV",
                      NCCL_DEBUG_FILE=str(args.output.resolve() / f"nccl-rank{rank}.log"))
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
    members = types.Placement.from_mesh(mesh)
    tests = cases(mesh)
    realization = Runtime([op for op, _, _ in tests])
    reports = []
    for op, shape, dtype in tests:
        places = types.Placement.from_mesh(op.participants)
        ranks = tuple(p.ids["gpu"] for p in places)
        pieces = tuple((p, torch.arange(math.prod(shape)).reshape(shape).to(dtype) + p.ids["gpu"])
                       for p in members)
        function = collective_function(op, shape, dtype)
        expected = evaluator.evaluate(function, core.ShardTensor(function.params[0].type, pieces))
        placement = types.Placement({"gpu": rank})
        source = dict((p.ids["gpu"], tensor) for p, tensor in pieces)[rank]
        local = source.to(device)
        if rank in ranks:
            if isinstance(op, sharding.Exchange) and not sum(op.counts[ranks.index(rank)]):
                local = None
            elif isinstance(op, sharding.Collective) and op.kind in {"broadcast", "scatter"} and rank != ranks[op.root]:
                local = None
        actual, result_shape = realization.apply(op, local, shape, device, dtype=dtype)
        if rank not in ranks:
            torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)
        elif placement in expected.placements:
            torch.testing.assert_close(actual.cpu(), expected.to_local(placement), rtol=0, atol=0)
            assert tuple(actual.shape) == result_shape and actual.dtype == dtype
        else:
            assert actual is None
        reports.append(dict(kind="exchange" if isinstance(op, sharding.Exchange) else op.kind,
                            dtype=str(dtype), input_shape=shape, output_shape=result_shape,
                            context=op.context))
    result = dict(status="passed", rank=rank, cases=len(tests), checks=reports,
                  ordered_ranks=[p.ids["gpu"] for p in members], torch=torch.__version__,
                  nccl=torch.cuda.nccl.version(), transport=os.environ.get("NCCL_NET", "native"),
                  trace=realization.trace)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"Passed {len(tests)} primitive cases on all eight ranks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("native", "torch"))
    args, remaining = parser.parse_known_args()
    (native if args.backend == "native" else reference)(remaining)
