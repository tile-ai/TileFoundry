"""dst = lhs @ rhs + acc on SIMD-group registers (simdgroup_multiply_accumulate)."""

from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry_apple.metal.ir.tir import evaluation, layout


@expr.register_op(dialect="T", category="nn", namespace="metal", name="mma")
class Mma(expr.Op):
    """dst = lhs @ rhs + acc; acc may be dst (simdgroup_multiply_accumulate)."""

    capability = expr.OpCapability("simdgroup_multiply_accumulate")
    execution_mesh = layout.SIMDGROUPS

    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.fragment())
    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())
    acc = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())


@typeinfer.register(Mma)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Mma)
def _(call, ctx):
    lhs, rhs, dst = access.matmul_relations(
        ctx.type_of(call.args[1]).shape, ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return dst, lhs, rhs, dst, dst


@evaluator.register(Mma)
def _(ctx):
    return evaluation.write(ctx, ctx.args[1].data @ ctx.args[2].data + ctx.args[3].data)


__all__ = ["Mma"]
