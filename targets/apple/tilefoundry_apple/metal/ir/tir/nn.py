"""8x8 matrix products on SIMD-group registers."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr
from tilefoundry.ir.hir import nn

from .layout import MATRIX_FLOATS, SIMDGROUPS, fragment, one_of

R, W = expr.MemoryEffect.READ, expr.MemoryEffect.WRITE


@expr.register_op(dialect="T", category="nn", namespace="metal", name="matmul")
class MatMul(expr.Op):
    """dst = lhs @ rhs (simdgroup_multiply)."""

    capability = expr.OpCapability("simdgroup_multiply")
    execution_mesh = SIMDGROUPS
    hir_matches = (
        schedule.HirMatch(nn.MatMul, when={"a_layout": "MK", "b_layout": "KN"},
                          operands={"dst": "result", "lhs": "lhs", "rhs": "rhs"}),
    )

    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=W, pattern=fragment())
    lhs = expr.ParamDef(kind="input", effect=R, pattern=fragment())
    rhs = expr.ParamDef(kind="input", effect=R, pattern=fragment())


@expr.register_op(dialect="T", category="nn", namespace="metal", name="mma")
class Mma(expr.Op):
    """dst = lhs @ rhs + acc; acc may be dst (simdgroup_multiply_accumulate)."""

    capability = expr.OpCapability("simdgroup_multiply_accumulate")
    execution_mesh = SIMDGROUPS

    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=W, pattern=fragment())
    lhs = expr.ParamDef(kind="input", effect=R, pattern=fragment())
    rhs = expr.ParamDef(kind="input", effect=R, pattern=fragment())
    acc = expr.ParamDef(kind="input", effect=R, pattern=fragment())


__all__ = ["MatMul", "Mma"]
