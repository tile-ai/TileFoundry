"""The T.metal.prefix instruction, its kinds and its rules."""

from enum import Enum

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry_apple.metal.ir.tir import evaluation, layout


@expr.register_type(dialect="T", namespace="metal")
class PrefixKind(Enum):
    """The running combination Prefix computes."""

    SUM = "sum"
    PRODUCT = "product"


@expr.register_op(dialect="T", category="arith", namespace="metal", name="prefix")
class Prefix(expr.Op):
    """Running sum or product over the lanes of each group (simd_prefix_inclusive_sum, ...)."""

    capability = tuple(
        expr.OpCapability(f"simd_prefix_{bound}_{kind.value}")
        for bound in ("inclusive", "exclusive") for kind in PrefixKind
    )
    execution_mesh = layout.LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=PrefixKind,
                         pattern=layout.one_of("kind", PrefixKind), default=PrefixKind.SUM)
    inclusive = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.lanes())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.lanes())


@typeinfer.register(Prefix)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Prefix)
def _(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    bound = "k <= l" if call.target.inclusive else "k < l"
    src = access.Relation(isl.map(f"{{ [g, l] -> [g, k] : 0 <= k and {bound} }}"))
    return access.iterating(shape, (access.identity(2), src, access.identity(2)))


@evaluator.register(Prefix)
def _(ctx):
    data = evaluation.widened(ctx.args[1])
    total = data.cumsum(1) if ctx.op.kind is PrefixKind.SUM else data.cumprod(1)
    if ctx.op.inclusive:
        return evaluation.write(ctx, total)
    head = data[:, :1]
    first = utils.torch.zeros_like(head) if ctx.op.kind is PrefixKind.SUM else utils.torch.ones_like(head)
    return evaluation.write(ctx, utils.torch.cat((first, total[:, :-1]), 1))


__all__ = ["Prefix", "PrefixKind"]
