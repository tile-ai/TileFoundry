"""The T.metal.shuffle_fill instruction and its rules."""

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry_apple.metal.ir.tir import evaluation, layout

from .shuffle import ShuffleKind


@expr.register_op(dialect="T", category="arith", namespace="metal", name="shuffle_fill")
class ShuffleFill(expr.Op):
    """Shift each sub-vector of modulo lanes; vacated lanes read fill (simd_shuffle_and_fill_*)."""

    capability = tuple(expr.OpCapability(f"simd_shuffle_and_fill_{name}") for name in ("up", "down"))
    execution_mesh = layout.LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=ShuffleKind,
                         pattern=layout.one_of("kind", (ShuffleKind.UP, ShuffleKind.DOWN)), default=ShuffleKind.UP)
    modulo = expr.ParamDef(kind="attribute", annotation=int, default=32, pattern=pattern.Wildcard(
        "modulo", predicates=(pattern.Or(2, 4, 8, 16, 32), pattern.Wildcard("modulo") <= layout.W)))
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.lanes())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.lanes())
    fill = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.lanes())
    delta = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.is_scalar_tensor(dtype=expr.DType.u16))


@typeinfer.register(ShuffleFill)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(ShuffleFill)
def _(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    dst = access.identity(2)
    any_lane = access.Relation(isl.map(f"{{ [g, l] -> [g, k] : 0 <= k < {shape[1]} }}"), may_access=True)
    delta = access.Relation(isl.map("{ [g, l] -> [] }"))
    return access.iterating(shape, (dst, any_lane, any_lane, delta, dst))


@evaluator.register(ShuffleFill)
def _(ctx):
    data, fill = evaluation.widened(ctx.args[1]), evaluation.widened(ctx.args[2])
    delta = int(ctx.args[3].data)
    groups, width = data.shape
    modulo = ctx.op.modulo
    if not 0 <= delta <= modulo or modulo > width:
        raise evaluator.EvalError("shuffle_fill delta or modulo outside the group")
    lane = utils.torch.arange(width).expand(groups, width)
    start = lane - lane % modulo
    wanted = lane % modulo + (-delta if ctx.op.kind is ShuffleKind.UP else delta)
    inside = (wanted >= 0) & (wanted < modulo)
    return evaluation.write(ctx, utils.torch.where(
        inside, data.gather(1, start + wanted.clamp(0, modulo - 1)), fill.gather(1, start + wanted % modulo)))


__all__ = ["ShuffleFill"]
