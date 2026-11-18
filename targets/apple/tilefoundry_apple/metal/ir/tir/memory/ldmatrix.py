"""Load one 8x8 matrix per SIMD group from memory into registers (simdgroup_load)."""

import isl

from tilefoundry.interfaces import access, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import sharding
from tilefoundry_apple.metal.ir.tir import evaluation, layout

TRANSPOSED = access.Relation(isl.map("{ [g, m, n] -> [g, n, m] }"))


@expr.register_op(dialect="T", category="memory", namespace="metal", name="ldmatrix")
class LdMatrix(expr.Op):
    """Load one 8x8 matrix per SIMD group from memory (simdgroup_load)."""

    capability = expr.OpCapability("simdgroup_load")
    execution_mesh = layout.SIMDGROUPS
    hir_matches = (
        schedule.HirMatch(sharding.Reshard, attributes={"transpose": False}, operands={"dst": "result", "src": "x"}),
    )

    transpose = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.fragment())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.memory_matrices())


@typeinfer.register(LdMatrix)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(LdMatrix)
def _(call, ctx):
    shape = ctx.type_of(call.args[0]).shape
    plain = access.identity(3)
    read = TRANSPOSED if call.target.transpose else plain
    return access.iterating(shape, (plain, read, plain))


@evaluator.register(LdMatrix)
def _(ctx):
    source = ctx.args[1].data
    return evaluation.write(ctx, source.transpose(-2, -1) if ctx.op.transpose else source)


__all__ = ["LdMatrix"]
