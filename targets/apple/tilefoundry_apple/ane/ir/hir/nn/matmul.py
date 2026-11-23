"""Where an ANE MatMul leaves its result: in the on-chip buffer once an operand is there."""

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple.ane import target as ane_target


@typeinfer.register(nn.MatMul, target=ane_target.Target)
def ane_matmul(call, ctx) -> types.TensorType:
    lhs, rhs = (ctx.type_of(arg) for arg in call.args)
    shape, layout = nn.matmul_result_shape_and_layout(call, ctx)
    if expr.StorageKind.SMEM in (lhs.storage, rhs.storage):
        storage = expr.StorageKind.SMEM
    else:
        storage = typeinfer.resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return types.TensorType(
        shape=shape, dtype=nn.matmul_result_dtype(call.target, lhs), layout=layout, storage=storage
    )


__all__: list[str] = []
