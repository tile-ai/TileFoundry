"""SM90 tensor-map reduction from shared memory into an explicit destination."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils

from .copy_async_tensor import tma_box_pattern, tma_global_pattern, warp_scope


@expr.register_op(dialect="T", category="async", name="copy_reduce_async_tensor")
class CopyReduceAsyncTensor(expr.Op):
    """Add a logical partial sum to the global destination; hardware adds in L2."""

    capability = expr.OpCapability("cp.async.bulk.tensor")
    resource = "tma_engine"
    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.operand_tile(
            0, expr.StorageKind.SMEM, tma_box_pattern(pattern.dtype_place(0), allow_partial=True)
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(1, expr.StorageKind.GMEM, tma_global_pattern(pattern.dtype_place(1))),
    )
    kind = expr.ParamDef(kind="attribute", pattern="add", default="add")
    execution_mesh = expr.ParamDef(
        kind="attribute", annotation=types.Mesh, pattern=warp_scope(), optional=True, default=None
    )
    between = (
        pattern.SameConstraint("shape", "src", "dst"),
        pattern.SameConstraint("dtype", "src", "dst"),
        pattern.SameModesConstraint("src", "dst", allow_partial=True),
    )


@typeinfer.register(CopyReduceAsyncTensor)
def _(call, ctx):
    return types.UnitType()


access.register_relation(CopyReduceAsyncTensor)(access.identity_relations)


@evaluator.register(CopyReduceAsyncTensor)
def _(ctx):
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, ctx.args[1].data + ctx.args[0].data))


@cost.register(CopyReduceAsyncTensor)
def _(call, ctx):
    src, dst = (utils.tensor_bytes(ctx.local_type_of(arg)) for arg in call.args)
    return cost.Cost({}, (cost.TrafficBytes(read=src), cost.TrafficBytes(write=dst), cost.TrafficBytes()))


__all__ = ["CopyReduceAsyncTensor"]
