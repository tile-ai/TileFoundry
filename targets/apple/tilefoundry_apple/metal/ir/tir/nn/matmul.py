"""dst = lhs @ rhs on SIMD-group registers (simdgroup_multiply)."""

from tilefoundry.interfaces import access, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import nn
from tilefoundry_apple.metal.ir.tir import evaluation, layout


@expr.register_op(dialect="T", category="nn", namespace="metal", name="matmul")
class MatMul(expr.Op):
    """dst = lhs @ rhs (simdgroup_multiply)."""

    capability = expr.OpCapability("simdgroup_multiply")
    execution_mesh = layout.SIMDGROUPS
    hir_matches = (
        schedule.HirMatch(nn.MatMul, when={"a_layout": "MK", "b_layout": "KN"},
                          operands={"dst": "result", "lhs": "lhs", "rhs": "rhs"}),
    )

    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.MATRIX_FLOATS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.fragment())
    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.fragment())


@typeinfer.register(MatMul)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(MatMul)
def _(call, ctx):
    lhs, rhs, dst = access.matmul_relations(
        ctx.type_of(call.args[1]).shape, ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return dst, lhs, rhs, dst


@evaluator.register(MatMul)
def _(ctx):
    return evaluation.write(ctx, ctx.args[1].data @ ctx.args[2].data)


__all__ = ["MatMul"]
