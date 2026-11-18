"""Store one 8x8 matrix per SIMD group from registers to memory (simdgroup_store)."""

import isl

from tilefoundry.interfaces import access, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import sharding
from tilefoundry_apple.metal.ir.tir import evaluation, layout

TRANSPOSED = access.Relation(isl.map("{ [g, m, n] -> [g, n, m] }"))


@expr.register_op(dialect="T", category="memory", namespace="metal", name="stmatrix")
class StMatrix(expr.Op):
    """Store one 8x8 matrix per SIMD group to memory (simdgroup_store)."""

    capability = expr.OpCapability("simdgroup_store")
    execution_mesh = layout.SIMDGROUPS
    hir_matches = (
        schedule.HirMatch(sharding.Reshard, attributes={"transpose": False}, operands={"dst": "result", "src": "x"}),
    )

    transpose = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.memory_matrices())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())


@typeinfer.register(StMatrix)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(StMatrix)
def _(call, ctx):
    shape = ctx.type_of(call.args[0]).shape
    plain = access.identity(3)
    written = TRANSPOSED if call.target.transpose else plain
    return access.iterating(shape, (written, plain, written))


@evaluator.register(StMatrix)
def _(ctx):
    source = ctx.args[1].data
    return evaluation.write(ctx, source.transpose(-2, -1) if ctx.op.transpose else source)


__all__ = ["StMatrix"]
