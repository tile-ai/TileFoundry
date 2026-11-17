"""Type inference, access relations and torch evaluation of the Metal Ops."""

import functools
from types import MappingProxyType

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import types

from .memory import LdMatrix, StMatrix
from .nn import MatMul, Mma
from .simd import Ballot, Prefix, PrefixKind, Reduce, ReduceKind, Shuffle, ShuffleFill, ShuffleKind

torch = utils.torch
TRANSPOSED = access.Relation(isl.map("{ [g, m, n] -> [g, n, m] }"))


def statement(call, ctx) -> types.TupleType:
    """Every Metal instruction writes its destination operand and returns nothing."""
    return types.TupleType.Void


def write(ctx, result):
    destination = ctx.args[0]
    destination.data.copy_(result.to(destination.data.dtype))
    return destination


# ---- matrices -------------------------------------------------------------

def move_access(call, ctx):
    shape = ctx.type_of(call.args[0]).shape
    plain = access.identity(3)
    moved = TRANSPOSED if call.target.transpose else plain
    read, written = (moved, plain) if isinstance(call.target, LdMatrix) else (plain, moved)
    return access.iterating(shape, (written, read, written))


def product_access(call, ctx):
    lhs, rhs, dst = access.matmul_relations(
        ctx.type_of(call.args[1]).shape, ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return (dst, lhs, rhs, dst) if isinstance(call.target, MatMul) else (dst, lhs, rhs, dst, dst)


def move(ctx):
    source = ctx.args[1].data
    return write(ctx, source.transpose(-2, -1) if ctx.op.transpose else source)


def matmul(ctx):
    return write(ctx, ctx.args[1].data @ ctx.args[2].data)


def mma(ctx):
    return write(ctx, ctx.args[1].data @ ctx.args[2].data + ctx.args[3].data)


# ---- lanes ----------------------------------------------------------------

REDUCTIONS = MappingProxyType({
    ReduceKind.SUM: lambda x: x.sum(1, keepdim=True),
    ReduceKind.PRODUCT: lambda x: x.prod(1, keepdim=True),
    ReduceKind.MIN: lambda x: x.amin(1, keepdim=True),
    ReduceKind.MAX: lambda x: x.amax(1, keepdim=True),
    ReduceKind.ALL: lambda x: x.all(1, keepdim=True),
    ReduceKind.ANY: lambda x: x.any(1, keepdim=True),
    ReduceKind.AND: lambda x: functools.reduce(torch.bitwise_and, x.unbind(1))[:, None],
    ReduceKind.OR: lambda x: functools.reduce(torch.bitwise_or, x.unbind(1))[:, None],
    ReduceKind.XOR: lambda x: functools.reduce(torch.bitwise_xor, x.unbind(1))[:, None],
})


def widened(value):
    """Lane data torch can combine: u16/u32 read as int64; write() narrows back with wrap-around."""
    data = value.data
    return data.to(torch.int64) if data.dtype in (torch.uint16, torch.uint32) else data


def partial(type_) -> bool:
    return any(isinstance(attr, types.Partial) for attr in getattr(type_.layout, "attrs", ()))


def reduce(ctx):
    source = ctx.args[1]
    if partial(source.type):
        return write(ctx, source.tensor.to_global().tensor)
    return write(ctx, REDUCTIONS[ctx.op.kind](widened(source)))


def prefix(ctx):
    data = widened(ctx.args[1])
    total = data.cumsum(1) if ctx.op.kind is PrefixKind.SUM else data.cumprod(1)
    if ctx.op.inclusive:
        return write(ctx, total)
    first = torch.zeros_like(data[:, :1]) if ctx.op.kind is PrefixKind.SUM else torch.ones_like(data[:, :1])
    return write(ctx, torch.cat((first, total[:, :-1]), 1))


def sources(kind, lane, amount, width):
    """The lane each lane reads; a lane with no source reads itself."""
    if kind in (ShuffleKind.INDEX, ShuffleKind.BROADCAST):
        return amount.expand_as(lane)
    if kind is ShuffleKind.XOR:
        return lane ^ amount
    wanted = lane - amount if kind in (ShuffleKind.UP, ShuffleKind.ROTATE_UP) else lane + amount
    if kind in (ShuffleKind.ROTATE_UP, ShuffleKind.ROTATE_DOWN):
        return wanted % width
    return torch.where((wanted < 0) | (wanted >= width), lane, wanted)


def shuffle(ctx):
    data, amount = widened(ctx.args[1]), ctx.args[2].data.to(torch.int64)
    groups, width = data.shape
    lane = torch.arange(width).expand(groups, width)
    source = sources(ctx.op.kind, lane, amount, width)
    if bool(((source < 0) | (source >= width)).any()):
        raise evaluator.EvalError("shuffle selector outside the group")
    return write(ctx, data.gather(1, source))


def shuffle_fill(ctx):
    data, fill, delta = widened(ctx.args[1]), widened(ctx.args[2]), int(ctx.args[3].data)
    groups, width = data.shape
    modulo = ctx.op.modulo
    if not 0 <= delta <= modulo or modulo > width:
        raise evaluator.EvalError("shuffle_fill delta or modulo outside the group")
    lane = torch.arange(width).expand(groups, width)
    start = lane - lane % modulo
    wanted = lane % modulo + (-delta if ctx.op.kind is ShuffleKind.UP else delta)
    inside = (wanted >= 0) & (wanted < modulo)
    return write(ctx, torch.where(inside, data.gather(1, start + wanted.clamp(0, modulo - 1)),
                                  fill.gather(1, start + wanted % modulo)))


def ballot(ctx):
    bits = ctx.args[1].data.to(torch.int64)
    return write(ctx, (bits << torch.arange(bits.shape[1])).sum(1, keepdim=True))


def to_group_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    if partial(ctx.type_of(call.args[1])):
        return access.iterating(shape, (access.identity(2),) * 3)
    dst = access.Relation(isl.map("{ [g, l] -> [g, 0] }"))
    return access.iterating(shape, (dst, access.identity(2), dst))


def prefix_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    bound = "k <= l" if call.target.inclusive else "k < l"
    src = access.Relation(isl.map(f"{{ [g, l] -> [g, k] : 0 <= k and {bound} }}"))
    return access.iterating(shape, (access.identity(2), src, access.identity(2)))


def shuffle_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    dst = access.identity(2)
    any_lane = access.Relation(isl.map(f"{{ [g, l] -> [g, k] : 0 <= k < {shape[1]} }}"), may_access=True)
    selector = access.identity(2) if call.target.kind is ShuffleKind.INDEX else access.Relation(isl.map("{ [g, l] -> [] }"))
    reads = (any_lane, selector) if isinstance(call.target, Shuffle) else (any_lane, any_lane, selector)
    return access.iterating(shape, (dst, *reads, dst))


SEMANTICS = (
    (LdMatrix, move_access, move), (StMatrix, move_access, move),
    (MatMul, product_access, matmul), (Mma, product_access, mma),
    (Reduce, to_group_access, reduce), (Ballot, to_group_access, ballot),
    (Prefix, prefix_access, prefix), (Shuffle, shuffle_access, shuffle), (ShuffleFill, shuffle_access, shuffle_fill),
)
for op, relation, evaluate in SEMANTICS:
    typeinfer.register(op)(statement)
    access.register_relation(op)(relation)
    evaluator.register(op)(evaluate)

__all__ = ["SEMANTICS"]
