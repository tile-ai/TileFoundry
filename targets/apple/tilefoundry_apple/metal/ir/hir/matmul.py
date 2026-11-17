"""Where a Metal MatMul leaves its result: in SIMD-group registers once an operand is on chip.

Operands already in the SIMD-group fragment make one cooperative product per
group, not a product of K-aligned shards: the result keeps the same fragment.
"""

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple.metal import target as metal_target
from tilefoundry_apple.metal.ir import tir

ON_CHIP = (expr.StorageKind.SMEM, expr.StorageKind.RMEM)


def fragments(call, lhs: types.TensorType, rhs: types.TensorType) -> bool:
    """Whether both operands are SIMD-group fragments of one mesh in MK x KN order."""
    if (call.target.a_layout, call.target.b_layout) != ("MK", "KN"):
        return False
    matcher = pattern.Matcher()
    declared = tir.layout.fragment()
    return (
        matcher.match(declared, lhs)
        and matcher.match(declared, rhs)
        and matcher.solve() is True
        and lhs.layout.mesh == rhs.layout.mesh
    )


@typeinfer.register(nn.MatMul, target=metal_target.Target)
def metal_matmul(call, ctx) -> types.TensorType:
    lhs, rhs = (ctx.type_of(arg) for arg in call.args)
    dtype = nn.matmul_result_dtype(call.target, lhs)
    if fragments(call, lhs, rhs):
        return types.TensorType(shape=lhs.shape, dtype=dtype, layout=lhs.layout, storage=expr.StorageKind.RMEM)
    shape, layout = nn.matmul_result_shape_and_layout(call, ctx)
    on_chip = lhs.storage in ON_CHIP or rhs.storage in ON_CHIP
    storage = expr.StorageKind.RMEM if on_chip else typeinfer.resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return types.TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


__all__: list[str] = []
