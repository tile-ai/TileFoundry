"""Matrix load and store between memory and SIMD-group registers."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import sharding

from .layout import MATRIX_FLOATS, SIMDGROUPS, fragment, memory_matrices, one_of

R, W = expr.MemoryEffect.READ, expr.MemoryEffect.WRITE
MOVES = (schedule.HirMatch(sharding.Reshard, attributes={"transpose": False}, operands={"dst": "result", "src": "x"}),)


@expr.register_op(dialect="T", category="memory", namespace="metal", name="ldmatrix")
class LdMatrix(expr.Op):
    """Load one 8x8 matrix per SIMD group from memory (simdgroup_load)."""

    capability = expr.OpCapability("simdgroup_load")
    execution_mesh = SIMDGROUPS
    hir_matches = MOVES

    transpose = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=W, pattern=fragment())
    src = expr.ParamDef(kind="input", effect=R, pattern=memory_matrices())


@expr.register_op(dialect="T", category="memory", namespace="metal", name="stmatrix")
class StMatrix(expr.Op):
    """Store one 8x8 matrix per SIMD group to memory (simdgroup_store)."""

    capability = expr.OpCapability("simdgroup_store")
    execution_mesh = SIMDGROUPS
    hir_matches = MOVES

    transpose = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=W, pattern=memory_matrices())
    src = expr.ParamDef(kind="input", effect=R, pattern=fragment())


__all__ = ["LdMatrix", "StMatrix"]
