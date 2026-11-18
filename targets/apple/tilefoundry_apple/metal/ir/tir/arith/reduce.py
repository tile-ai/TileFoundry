"""The T.metal.reduce instruction, its kinds and its rules."""

import functools
from enum import Enum
from types import MappingProxyType

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import sharding
from tilefoundry_apple.metal.ir.tir import evaluation, layout


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


BITWISE = (ReduceKind.AND, ReduceKind.OR, ReduceKind.XOR)
VOTES = (ReduceKind.ALL, ReduceKind.ANY)
REDUCE_DTYPES = pattern.Switch("kind", {
    kind: expr.DType.bool if kind in VOTES else pattern.Or(*(layout.LANE_INTS if kind in BITWISE else layout.LANE_NUMBERS))
    for kind in ReduceKind
})
PARTIAL = MappingProxyType({
    kind: pattern.ShardLayout(
        pattern.Layout((layout.G, 1), (0, 0)), (types.Split(0), types.Partial(kind.value)), layout.LANE_GROUPS)
    for kind in (ReduceKind.SUM, ReduceKind.MIN, ReduceKind.MAX)
})
REDUCTIONS = MappingProxyType({
    ReduceKind.SUM: lambda x: x.sum(1, keepdim=True),
    ReduceKind.PRODUCT: lambda x: x.prod(1, keepdim=True),
    ReduceKind.MIN: lambda x: x.amin(1, keepdim=True),
    ReduceKind.MAX: lambda x: x.amax(1, keepdim=True),
    ReduceKind.ALL: lambda x: x.all(1, keepdim=True),
    ReduceKind.ANY: lambda x: x.any(1, keepdim=True),
    ReduceKind.AND: lambda x: functools.reduce(utils.torch.bitwise_and, x.unbind(1))[:, None],
    ReduceKind.OR: lambda x: functools.reduce(utils.torch.bitwise_or, x.unbind(1))[:, None],
    ReduceKind.XOR: lambda x: functools.reduce(utils.torch.bitwise_xor, x.unbind(1))[:, None],
})


@expr.register_op(dialect="T", category="arith", namespace="metal", name="reduce")
class Reduce(expr.Op):
    """Combine the lanes of each group and give every lane the result (simd_sum, simd_min, ...).

    src holds one scalar per lane, or the partial values of a reduction the HIR
    already split per lane (PARTIAL), which a Reshard to Broadcast resolves.
    """

    capability = tuple(expr.OpCapability(f"simd_{kind.value}") for kind in ReduceKind)
    execution_mesh = layout.LANE_GROUPS
    hir_matches = tuple(
        schedule.HirMatch(sharding.Reshard, attributes={"kind": kind}, operands={"dst": "result", "src": "x"})
        for kind in PARTIAL
    )

    kind = expr.ParamDef(kind="attribute", annotation=ReduceKind,
                         pattern=layout.one_of("kind", ReduceKind), default=ReduceKind.SUM)
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType,
                          pattern=pattern.Wildcard("dtype", predicates=(REDUCE_DTYPES,)), default=expr.DType.f32)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.shared())
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.Tensor(
        shape=(layout.G, pattern.Or(layout.W, 1)), dtype=pattern.Wildcard("dtype"), storage=expr.StorageKind.RMEM,
        layout=pattern.Or(layout.LANES, pattern.Switch("kind", PARTIAL))))


def partial(type_) -> bool:
    """Whether *type_* holds per-lane partial values of a reduction."""
    return any(isinstance(attr, types.Partial) for attr in getattr(type_.layout, "attrs", ()))


@typeinfer.register(Reduce)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Reduce)
def _(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    if partial(ctx.type_of(call.args[1])):
        return access.iterating(shape, (access.identity(2),) * 3)
    dst = access.Relation(isl.map("{ [g, l] -> [g, 0] }"))
    return access.iterating(shape, (dst, access.identity(2), dst))


@evaluator.register(Reduce)
def _(ctx):
    source = ctx.args[1]
    if partial(source.type):
        return evaluation.write(ctx, source.tensor.to_global().tensor)
    return evaluation.write(ctx, REDUCTIONS[ctx.op.kind](evaluation.widened(source)))


__all__ = ["Reduce", "ReduceKind"]
