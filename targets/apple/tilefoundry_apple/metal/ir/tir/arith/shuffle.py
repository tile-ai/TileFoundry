"""The T.metal.shuffle instruction, its kinds and its rules."""

from enum import Enum

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry_apple.metal.ir.tir import evaluation, layout


@expr.register_type(dialect="T", namespace="metal")
class ShuffleKind(Enum):
    """Which lane each lane reads in Shuffle and ShuffleFill."""

    INDEX = "index"
    BROADCAST = "broadcast"
    XOR = "xor"
    UP = "up"
    DOWN = "down"
    ROTATE_UP = "rotate_up"
    ROTATE_DOWN = "rotate_down"


@expr.register_op(dialect="T", category="arith", namespace="metal", name="shuffle")
class Shuffle(expr.Op):
    """Each lane reads another lane of its group (simd_shuffle, simd_broadcast, ...).

    With kind INDEX, sel names one source lane per lane; every other kind takes
    one u16 for the whole call.
    """

    capability = tuple(
        expr.OpCapability(f"simd_{name}")
        for name in ("shuffle", "broadcast", "shuffle_xor", "shuffle_up", "shuffle_down",
                     "shuffle_rotate_up", "shuffle_rotate_down")
    )
    execution_mesh = layout.LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=ShuffleKind,
                         pattern=layout.one_of("kind", ShuffleKind), default=ShuffleKind.INDEX)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=layout.one_of("dtype", layout.LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.lanes())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.lanes())
    sel = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.Switch("kind", {
        kind: layout.lanes(expr.DType.u16) if kind is ShuffleKind.INDEX else pattern.is_scalar_tensor(dtype=expr.DType.u16)
        for kind in ShuffleKind
    }))


def sources(kind, lane, amount, width):
    """The lane each lane reads; a lane with no source reads itself."""
    if kind in (ShuffleKind.INDEX, ShuffleKind.BROADCAST):
        return amount.expand_as(lane)
    if kind is ShuffleKind.XOR:
        return lane ^ amount
    wanted = lane - amount if kind in (ShuffleKind.UP, ShuffleKind.ROTATE_UP) else lane + amount
    if kind in (ShuffleKind.ROTATE_UP, ShuffleKind.ROTATE_DOWN):
        return wanted % width
    return utils.torch.where((wanted < 0) | (wanted >= width), lane, wanted)


@typeinfer.register(Shuffle)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Shuffle)
def _(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    dst = access.identity(2)
    any_lane = access.Relation(isl.map(f"{{ [g, l] -> [g, k] : 0 <= k < {shape[1]} }}"), may_access=True)
    selector = access.identity(2) if call.target.kind is ShuffleKind.INDEX else access.Relation(isl.map("{ [g, l] -> [] }"))
    return access.iterating(shape, (dst, any_lane, selector, dst))


@evaluator.register(Shuffle)
def _(ctx):
    data, amount = evaluation.widened(ctx.args[1]), ctx.args[2].data.to(utils.torch.int64)
    groups, width = data.shape
    lane = utils.torch.arange(width).expand(groups, width)
    source = sources(ctx.op.kind, lane, amount, width)
    if bool(((source < 0) | (source >= width)).any()):
        raise evaluator.EvalError("shuffle selector outside the group")
    return evaluation.write(ctx, data.gather(1, source))


__all__ = ["Shuffle", "ShuffleKind"]
