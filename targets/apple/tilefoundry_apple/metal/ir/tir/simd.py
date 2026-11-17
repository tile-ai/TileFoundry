"""SIMD-group and quad-group functions: one scalar per lane, groups of 32 or 4 lanes."""

from enum import Enum
from types import MappingProxyType

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import sharding

from .layout import LANE_GROUPS, LANE_INTS, LANE_NUMBERS, LANES, G, W, lanes, one_of, shared

R, WRITE = expr.MemoryEffect.READ, expr.MemoryEffect.WRITE


@expr.register_type(dialect="T", namespace="metal")
class ReduceKind(Enum):
    """How Reduce combines the lanes of a group."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AND = "and"
    OR = "or"
    XOR = "xor"
    ALL = "all"
    ANY = "any"


@expr.register_type(dialect="T", namespace="metal")
class PrefixKind(Enum):
    """The running combination Prefix computes."""

    SUM = "sum"
    PRODUCT = "product"


@expr.register_type(dialect="T", namespace="metal")
class ShuffleKind(Enum):
    """Which lane each lane reads in Shuffle and ShuffleFill."""

    INDEX = "index"            # simd_shuffle: each lane names its source lane
    BROADCAST = "broadcast"    # simd_broadcast: every lane reads one lane
    XOR = "xor"
    UP = "up"
    DOWN = "down"
    ROTATE_UP = "rotate_up"
    ROTATE_DOWN = "rotate_down"


BITWISE = (ReduceKind.AND, ReduceKind.OR, ReduceKind.XOR)
VOTES = (ReduceKind.ALL, ReduceKind.ANY)
REDUCE_DTYPES = pattern.Switch("kind", {
    kind: expr.DType.bool if kind in VOTES else pattern.Or(*(LANE_INTS if kind in BITWISE else LANE_NUMBERS))
    for kind in ReduceKind
})
# A reduction the HIR already split per lane: every lane holds a partial value of its group.
PARTIAL = MappingProxyType({
    kind: pattern.ShardLayout(pattern.Layout((G, 1), (0, 0)), (types.Split(0), types.Partial(kind.value)), LANE_GROUPS)
    for kind in (ReduceKind.SUM, ReduceKind.MIN, ReduceKind.MAX)
})


def native(prefix: str, names) -> tuple[expr.OpCapability, ...]:
    return tuple(expr.OpCapability(f"{prefix}_{name}") for name in names)


@expr.register_op(dialect="T", category="arith", namespace="metal", name="reduce")
class Reduce(expr.Op):
    """Combine the lanes of each group and give every lane the result (simd_sum, simd_min, ...)."""

    capability = native("simd", (kind.value for kind in ReduceKind))
    execution_mesh = LANE_GROUPS
    hir_matches = tuple(
        schedule.HirMatch(sharding.Reshard, attributes={"kind": kind}, operands={"dst": "result", "src": "x"})
        for kind in PARTIAL
    )

    kind = expr.ParamDef(kind="attribute", annotation=ReduceKind, pattern=one_of("kind", ReduceKind), default=ReduceKind.SUM)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=pattern.Wildcard("dtype", predicates=(REDUCE_DTYPES,)), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=WRITE, pattern=shared())
    # One scalar per lane, or the per-lane partial values a Reshard to Broadcast resolves
    src = expr.ParamDef(kind="input", effect=R, pattern=pattern.Tensor(
        shape=(G, pattern.Or(W, 1)), dtype=pattern.Wildcard("dtype"), storage=expr.StorageKind.RMEM,
        layout=pattern.Or(LANES, pattern.Switch("kind", PARTIAL))))


@expr.register_op(dialect="T", category="arith", namespace="metal", name="prefix")
class Prefix(expr.Op):
    """Running sum or product over the lanes of each group (simd_prefix_inclusive_sum, ...)."""

    capability = native("simd_prefix", (f"{bound}_{kind.value}" for bound in ("inclusive", "exclusive") for kind in PrefixKind))
    execution_mesh = LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=PrefixKind, pattern=one_of("kind", PrefixKind), default=PrefixKind.SUM)
    inclusive = expr.ParamDef(kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=WRITE, pattern=lanes())
    src = expr.ParamDef(kind="input", effect=R, pattern=lanes())


@expr.register_op(dialect="T", category="arith", namespace="metal", name="shuffle")
class Shuffle(expr.Op):
    """Each lane reads another lane of its group (simd_shuffle, simd_shuffle_up, simd_broadcast, ...)."""

    capability = native("simd", ("shuffle", "broadcast", "shuffle_xor", "shuffle_up", "shuffle_down",
                                 "shuffle_rotate_up", "shuffle_rotate_down"))
    execution_mesh = LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=ShuffleKind, pattern=one_of("kind", ShuffleKind), default=ShuffleKind.INDEX)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=WRITE, pattern=lanes())
    src = expr.ParamDef(kind="input", effect=R, pattern=lanes())
    # INDEX: one source lane per lane. Every other kind: one u16 for the whole call.
    sel = expr.ParamDef(kind="input", effect=R, pattern=pattern.Switch("kind", {
        kind: lanes(expr.DType.u16) if kind is ShuffleKind.INDEX else pattern.is_scalar_tensor(dtype=expr.DType.u16)
        for kind in ShuffleKind
    }))


@expr.register_op(dialect="T", category="arith", namespace="metal", name="shuffle_fill")
class ShuffleFill(expr.Op):
    """Shift each sub-vector of modulo lanes and take the vacated lanes from fill (simd_shuffle_and_fill_up/down)."""

    capability = native("simd_shuffle_and_fill", ("up", "down"))
    execution_mesh = LANE_GROUPS

    kind = expr.ParamDef(kind="attribute", annotation=ShuffleKind,
                         pattern=one_of("kind", (ShuffleKind.UP, ShuffleKind.DOWN)), default=ShuffleKind.UP)
    modulo = expr.ParamDef(kind="attribute", annotation=int, default=32, pattern=pattern.Wildcard(
        "modulo", predicates=(pattern.Or(2, 4, 8, 16, 32), pattern.Wildcard("modulo") <= W)))
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, pattern=one_of("dtype", LANE_NUMBERS), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=WRITE, pattern=lanes())
    src = expr.ParamDef(kind="input", effect=R, pattern=lanes())
    fill = expr.ParamDef(kind="input", effect=R, pattern=lanes())
    delta = expr.ParamDef(kind="input", effect=R, pattern=pattern.is_scalar_tensor(dtype=expr.DType.u16))


@expr.register_op(dialect="T", category="arith", namespace="metal", name="ballot")
class Ballot(expr.Op):
    """One bit per lane of a bool, as a u32 every lane of the group reads (simd_ballot)."""

    capability = expr.OpCapability("simd_ballot")
    execution_mesh = LANE_GROUPS

    dst = expr.ParamDef(kind="input", effect=WRITE, pattern=shared(expr.DType.u32))
    src = expr.ParamDef(kind="input", effect=R, pattern=lanes(expr.DType.bool))


__all__ = ["Ballot", "Prefix", "PrefixKind", "Reduce", "ReduceKind", "Shuffle", "ShuffleFill", "ShuffleKind"]
