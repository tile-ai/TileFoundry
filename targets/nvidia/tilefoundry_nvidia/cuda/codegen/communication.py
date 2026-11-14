"""Lower generic host communication statements to NCCL and CUDA runtime calls."""

from importlib.resources import files
from math import prod
from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.analysis import communication
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as comm

__all__ = ["NCCL"]

HEADER = '#include <tilefoundry/runtime/cuda/nccl.h>'
PREFIX = "tilefoundry::nccl::"
DTYPES = MappingProxyType({"f16": "ncclFloat16", "bf16": "ncclBfloat16", "f32": "ncclFloat32",
          "f64": "ncclFloat64", "i8": "ncclInt8", "u8": "ncclUint8",
          "i32": "ncclInt32", "u32": "ncclUint32", "i64": "ncclInt64"})
KINDS = MappingProxyType({name: i for i, name in enumerate(("broadcast", "reduce", "all_reduce", "gather", "scatter", "all_gather", "reduce_scatter", "exchange"))})


def emit_resources(entry, ctx):
    """Validate NCCL operands and reject undeclared runtime aliases before issuing work."""
    lines = [f"{PREFIX}Session tilefoundry_comm;"]
    tensors = [param for param in entry.params if isinstance(param.type, types.TensorType)]
    regions = communication.buffer_regions(entry)
    operands = {
        id(arg)
        for stmt in communication.host_statements(entry.body)
        if isinstance(stmt.callable, comm.CommEnqueue)
        for arg in (stmt.args[0], stmt.args[1], *stmt.args[4:])
    }
    for param in tensors:
        if id(param) not in operands:
            continue
        ty = param.type
        if ty.storage not in {expr.StorageKind.GMEM, expr.StorageKind.HOST}:
            raise ValueError("NCCL host operands must declare GMEM or HOST storage")
        if ty.dtype.name not in DTYPES:
            raise ValueError(f"NCCL does not support dtype {ty.dtype.name}")
        shape = ", ".join(map(str, ty.shape))
        code = "kDLFloat" if isinstance(ty.dtype, expr.FloatDType) else "kDLInt" if ty.dtype.signed else "kDLUInt"
        if ty.dtype == expr.DType.bf16:
            code = "kDLBfloat"
        device = "kDLCUDA" if ty.storage == expr.StorageKind.GMEM else "kDLCPU"
        lines.append(f"{PREFIX}tensor({param.name}, {{{shape}}}, {code}, {ty.dtype.bit_width}, {device});")
    for i, first in enumerate(tensors):
        for second in tensors[i + 1:]:
            a = regions.get(first.name, communication.BufferRegion(first.name))
            b = regions.get(second.name, communication.BufferRegion(second.name))
            if a.allocation == b.allocation:
                lines.append(
                    f"if ({first.name}.device().device_type != {second.name}.device().device_type || "
                    f"{first.name}.device().device_id != {second.name}.device().device_id || "
                    f"reinterpret_cast<uintptr_t>({first.name}.data_ptr()) + {b.offset} != "
                    f"reinterpret_cast<uintptr_t>({second.name}.data_ptr()) + {a.offset}) "
                    'throw std::runtime_error("communication allocation binding mismatch");'
                )
            else:
                lines.append(f"{PREFIX}disjoint({first.name}, {second.name});")
    handles = [param for param in entry.params if isinstance(param.type, types.CommunicationHandleType)]
    for i, first in enumerate(handles):
        if first.type.kind != "stream":
            lines.append(f'if (!{first.name}) throw std::runtime_error("null communication handle");')
        for second in handles[i + 1:]:
            if first.type.kind == second.type.kind:
                lines.append(f'if ({first.name} == {second.name}) throw std::runtime_error("undeclared resource alias");')
    return HEADER, lines


def local_rank(operation, entry, ctx):
    ids = {sig.topology_level: sig.name for sig in ctx.signature_of(entry).leading}
    members = types.Placement.from_mesh(operation.participants)
    alternatives = []
    for rank, member in enumerate(members):
        if any(key not in ids for key in member.ids):
            raise ValueError("NCCL communication requires host program IDs for every participant topology")
        condition = " && ".join(f"{ids[key]} == {value}" for key, value in member.ids.items())
        alternatives.append(f"({condition}) ? {rank} : ")
    return "(" + "".join(alternatives) + "-1)"


def enqueue(entry, stmt, ctx):
    semantic = comm.check_enqueue(stmt.callable, stmt.args).target
    if not isinstance(semantic, (sharding.Collective, sharding.Exchange)):
        raise ValueError("NCCL lowering requires Collective or Exchange semantics")
    src, dst, communicator, stream, *metadata = stmt.args
    if src.type.storage != expr.StorageKind.GMEM or dst.type.storage != expr.StorageKind.GMEM:
        raise ValueError("NCCL payloads require GPU memory")
    dtype = DTYPES[src.type.dtype.name]
    n = len(types.Placement.from_mesh(semantic.participants))
    rank = local_rank(semantic, entry, ctx)
    counts = "{}"
    if isinstance(semantic, sharding.Collective):
        kind, root = KINDS[semantic.kind], semantic.root or 0
        reduction = {None: "ncclSum", "sum": "ncclSum", "avg": "ncclAvg", "max": "ncclMax"}[semantic.reduction]
        axis = semantic.axis % len(src.type.shape) if src.type.shape else 0
        outer = prod(src.type.shape[:axis])
    else:
        kind, root, reduction, outer = KINDS["exchange"], 0, "ncclSum", 1
        if semantic.counts is not None:
            counts = "{" + ", ".join(str(count) for row in semantic.counts for count in row) + "}"
        else:
            counts = f"{PREFIX}counts({metadata[0].name}, {n}, {PREFIX}stream({stream.name}))"
    return [f"tilefoundry_comm.enqueue({{{kind}, {src.name}.data_ptr(), {dst.name}.data_ptr(), "
            f"{prod(src.type.shape)}, {prod(dst.type.shape)}, {outer}, {src.type.dtype.bit_width // 8}, {dtype}, {reduction}, "
            f"{root}, {n}, {rank}, {PREFIX}communicator({communicator.name}), {PREFIX}stream({stream.name}), {counts}}});"]


def emit_communication(entry, stmt, ctx):
    """Keep grouped issue separate from CUDA event and stream completion."""
    op, args = stmt.callable, [arg.name for arg in stmt.args]
    if isinstance(op, comm.CommEnqueue):
        return HEADER, enqueue(entry, stmt, ctx)
    if isinstance(op, comm.CommGroupStart):
        return HEADER, ["tilefoundry_comm.start();"]
    if isinstance(op, comm.CommGroupEnd):
        return HEADER, ["tilefoundry_comm.end();"]
    if isinstance(op, comm.ExportEvent):
        return HEADER, []
    if isinstance(op, comm.EventRecord):
        call = f"cudaEventRecord({PREFIX}event({args[0]}), {PREFIX}stream({args[1]}))"
    elif isinstance(op, comm.StreamWaitEvent):
        call = f"cudaStreamWaitEvent({PREFIX}stream({args[0]}), {PREFIX}event({args[1]}), 0)"
    elif isinstance(op, comm.EventSynchronize):
        call = f"cudaEventSynchronize({PREFIX}event({args[0]}))"
    elif isinstance(op, comm.StreamSynchronize):
        call = f"cudaStreamSynchronize({PREFIX}stream({args[0]}))"
    else:
        raise ValueError(f"unsupported communication statement {type(op).__name__}")
    return HEADER, [f"{PREFIX}check({call});"]


class NCCL(codegen.Communication):
    """NCCL transfers with CUDA stream/event completion and native resources."""

    def emit_resources(self, entry, ctx):
        """Validate tensors and native resource identities before issuing work."""
        return emit_resources(entry, ctx)

    def emit_statement(self, entry, stmt, ctx):
        """Emit NCCL issue or CUDA completion for a checked generic statement."""
        return emit_communication(entry, stmt, ctx)

    def link(self, project):
        """Link NCCL only when this provider emitted a host protocol."""
        project.include(files(__package__).joinpath("include"))
        project.append(
            "find_package(CUDAToolkit REQUIRED)\n"
            "find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)\n"
            "find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 PATH_SUFFIXES lib lib64 REQUIRED)\n"
            "target_include_directories(${PROJECT_NAME} PRIVATE ${NCCL_INCLUDE_DIR})\n"
            "target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart ${NCCL_LIBRARY})\n"
        )
