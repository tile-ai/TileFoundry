"""Ready-made boundaries: identity, broadcast, contraction, gather, view and state."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

import isl

from tilefoundry.ir import dim, expr, types

from .iteration import iterating
from .relation import Relation
from .window import reached_at


def identity(rank: int) -> Relation:
    """Identity boundary for a tensor of *rank*: each element read where it is."""
    dims = ", ".join(f"d{index}" for index in range(rank))
    return Relation(isl.map(f"{{ [{dims}] -> [{dims}] }}" if rank else "{ [] -> [] }"))


def is_one(node) -> bool:
    """Return True for any shape entry that represents the literal 1.

    Shape entries can be either ``Constant(value=1)`` (the canonical IR
    form, produced by the parser / annotation lift) or a Python ``int``
    (produced ad-hoc by some typeinfer rules — Reduce, Slice, etc.). Both
    forms must broadcast against larger dims; restricting to ``Constant``
    only breaks the ``(1, N) ⊕ (1, 1)`` pattern that falls out of
    ``Reduce(..., keepdim=True)``.
    """
    if isinstance(node, expr.Constant) and node.value == 1:
        return True
    if isinstance(node, int) and not isinstance(node, bool) and node == 1:
        return True
    return False


def broadcast_shapes(a: tuple, b: tuple, *, raising: bool = True):
    """NumPy-style right-aligned broadcast on ``tuple[Expr, ...]``.

    The shorter shape is padded on the left with 1s, then dims combine
    pairwise (equal, or one is 1). For an incompatible pair: raises
    ``ValueError`` when *raising* (the default), else returns ``None``.
    """
    if a == b:
        return a
    n = max(len(a), len(b))
    ap = (1,) * (n - len(a)) + tuple(a)
    bp = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(ap, bp):
        if x == y:
            out.append(x)
        elif is_one(x):
            out.append(y)
        elif is_one(y):
            out.append(x)
        elif raising:
            raise ValueError(f"cannot broadcast shapes {a} and {b}")
        else:
            return None
    return tuple(out)


def broadcast_all(shapes: tuple[tuple, ...]) -> tuple:
    """The common right-aligned broadcast shape of these operands."""
    out_shape = shapes[0]
    for shape in shapes[1:]:
        out_shape = broadcast_shapes(out_shape, shape)
    return out_shape


def broadcast(result_shape: tuple, operand_shape: tuple) -> Relation:
    """Which coordinate of an operand a result coordinate reads.

    An operand of the result's own shape reads the coordinate it is at. A
    shorter one right-aligns, dropping the leading axes it does not have; an
    axis of extent one reads zero when the result axis grows beyond one, and
    otherwise reads its own coordinate. Both are stated as
    the relation they are.
    """
    rank = len(result_shape)
    dims = [f"d{index}" for index in range(rank)]
    offset = rank - len(operand_shape)
    reads = [
        "0"
        if is_one(operand_shape[index - offset]) and not is_one(result_shape[index])
        else dims[index]
        for index in range(offset, rank)
    ]
    domain = ", ".join(dims)
    if not reads:
        return Relation(isl.map(f"{{ [{domain}] -> [] }}" if rank else "{ [] -> [] }"))
    return Relation(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def operand_reads(
    shape: tuple,
    out_shape: tuple,
    out_axes: tuple[str, ...],
    inner: str,
    *,
    kept_axis: int,
    output_axis: int,
    contraction_axis: int,
) -> list[str]:
    """One coordinate per axis of an operand of the contraction.

    The two matrix axes are the contraction and the one the result keeps; which
    is which is the only difference between the two operands. Batch axes are
    right-aligned against the result's, and one the operand broadcasts reads its
    only coordinate rather than the result's.
    """
    rank = len(shape)
    kept_axis %= rank
    contraction_axis %= rank
    shift = len(out_shape) - rank
    reads: list[str] = []
    for axis in range(rank):
        if axis == contraction_axis:
            reads.append(inner)
        elif axis == kept_axis:
            reads.append(out_axes[output_axis])
        elif is_one(shape[axis]) and not is_one(out_shape[axis + shift]):
            reads.append("0")
        else:
            reads.append(out_axes[axis + shift])
    return reads


def gather_relations(
    source: types.TensorType, index: types.TensorType, axis: int
) -> tuple[Relation, Relation, Relation]:
    """Read source slices selected by index, then write the gathered result."""
    out_shape = (*source.shape[:axis], index.shape[0], *source.shape[axis + 1 :])
    rank = len(out_shape)
    carried = {position: f"d{position}" for position in range(rank)}
    return iterating(
        out_shape,
        (
            reached_at(rank, source, source, carried, free=(axis,)),
            reached_at(rank, index, index, {0: carried.get(axis, "0")}),
            identity(rank),
        ),
    )


def matmul_relations(
    lhs_shape: tuple,
    rhs_shape: tuple,
    axes: tuple[int, int, int, int],
) -> tuple[Relation, ...]:
    """Every coordinate of each operand a contraction reaches, read once.

    A product walks the axis it sums, so that axis is a coordinate this Op is
    asked by rather than something existential inside an image, and the result is
    accumulated over it. Reading an operand at the result's own coordinates would
    claim a shape it does not have the moment the summed and kept axes differ in
    extent. Which positions any of these coordinates are is the reader's
    question; the result's own extents follow from the operands.
    """
    a_m, a_k, b_n, b_k = axes
    batch = broadcast_shapes(tuple(lhs_shape[:-2]), tuple(rhs_shape[:-2]), raising=False)
    if batch is None:
        raise ValueError(
            f"MatMul batches {tuple(lhs_shape[:-2])} against "
            f"{tuple(rhs_shape[:-2])}, which do not broadcast"
        )
    out_shape = (*batch, lhs_shape[a_m], rhs_shape[b_n])
    summed = lhs_shape[a_k]
    out_axes = (*(f"d{index}" for index in range(len(batch))), "m", "n")
    dims = ", ".join((*out_axes, "k"))
    inner = "0" if is_one(summed) else "k"
    inputs = []
    for shape, kept_axis, output_axis, contraction_axis in (
        (tuple(lhs_shape), a_m, -2, a_k),
        (tuple(rhs_shape), b_n, -1, b_k),
    ):
        reads = operand_reads(
            shape,
            out_shape,
            out_axes,
            inner,
            kept_axis=kept_axis,
            output_axis=output_axis,
            contraction_axis=contraction_axis,
        )
        inputs.append(Relation(isl.map(f"{{ [{dims}] -> [{', '.join(reads)}] }}")))
    accumulates = ", ".join(out_axes)
    return iterating(
        (*out_shape, summed),
        (*inputs, Relation(isl.map(f"{{ [{dims}] -> [{accumulates}] }}"))),
    )


def unread(domain_rank: int, rank: int) -> Relation:
    """No coordinate of a rank-*rank* value is reached from a rank-*domain_rank* space."""
    domain = ", ".join(f"d{index}" for index in range(domain_rank))
    reads = ", ".join(f"i{index}" for index in range(rank))
    return Relation(isl.map(f"{{ [{domain}] -> [{reads}] : 1 = 0 }}"))


def readnone_relations(
    result_shape: Callable[[tuple], tuple],
) -> Callable[..., tuple[Relation, ...]]:
    """Handler for an Op that answers from its operands' Types, not their elements.

    A rank, a shape, a view of the same value: the answer is in the Types, so no
    coordinate is read and nothing crosses. *result_shape* derives the result's
    shape from the operand Types alone, because type inference asks this before
    the Call has a Type of its own.
    """

    def handler(call, ctx) -> tuple[Relation, ...]:
        type_list = tuple(ctx.type_of(arg) for arg in call.args)
        shape = tuple(result_shape(type_list))
        return iterating(
            shape,
            (
                *(unread(len(shape), len(getattr(type_, "shape", ()))) for type_ in type_list),
                unread(len(shape), len(shape)),
            ),
        )

    return handler


def linearized_view(out_shape: tuple, in_shape: tuple) -> Relation:
    """Where an output coordinate sits in a source of another shape.

    A reshape keeps the elements in the order they were in and renames the axes
    over them, so one flat index answers both sides. The domain is the output's,
    because that is the side a reader walks. An empty shape holds nothing, so
    nothing of it is anywhere in the source: the answer is an empty relation
    rather than a division by an axis of length zero.
    """
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0
        for extent in (*out_shape, *in_shape)
    ):
        raise ValueError(f"a view relabels a shape it can count: {out_shape!r} from {in_shape!r}")
    out_rank, in_rank = len(out_shape), len(in_shape)
    if 0 in out_shape or 0 in in_shape:
        dims = ", ".join(f"d{index}" for index in range(out_rank))
        reads = ", ".join("0" for _ in range(in_rank))
        return Relation(isl.map(f"{{ [{dims}] -> [{reads}] : 1 = 0 }}"))
    dims = [f"d{index}" for index in range(out_rank)]
    flat, stride = [], 1
    for index in reversed(range(out_rank)):
        flat.append(f"{dims[index]}" if stride == 1 else f"{stride} * {dims[index]}")
        stride *= out_shape[index]
    linear = " + ".join(reversed(flat)) if flat else "0"
    reads, stride = [], 1
    strides = []
    for extent in reversed(in_shape):
        strides.append(stride)
        stride *= extent
    for axis, step in zip(range(in_rank), reversed(strides)):
        term = f"({linear})" if step == 1 else f"floor(({linear}) / {step})"
        reads.append(term if in_shape[axis] == stride // step else f"({term}) mod {in_shape[axis]}")
    domain = ", ".join(dims)
    return Relation(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def identity_relations(call, ctx) -> tuple[Relation, ...]:
    """Walk the operands' broadcast domain and map each boundary into it.

    Structural operands whose shape is not inferred yet borrow the domain.
    Only operand types are read: HIR type inference derives its result here.
    Incompatible shapes expose an invalid relation instead of guessing a domain.
    """
    type_list = tuple(ctx.type_of(arg) for arg in call.args)
    shapes = tuple(tuple(ty.shape) if hasattr(ty, "shape") else None for ty in type_list)
    domain = reduce(broadcast_shapes, (shape for shape in shapes if shape is not None), ())

    def access(shape):
        return identity(len(domain)) if shape is None else broadcast(domain, shape)

    return iterating(domain, (*(access(shape) for shape in shapes), identity(len(domain))))


def state(type_, domain_rank=0):
    """Conservatively cover an opaque state's cells from one effect event."""
    domain = ", ".join(f"d{axis}" for axis in range(domain_rank))
    image = ", ".join(f"s{axis}" for axis in range(len(type_.shape)))
    relation = isl.map(f"{{ [{domain}] -> [{image}] }}")
    values = {}
    relation = relation.intersect_range(dim.shape_to_isl_set(type_.shape, values))
    return Relation(relation, values, may_access=True)


__all__ = [
    "broadcast",
    "broadcast_all",
    "broadcast_shapes",
    "gather_relations",
    "identity",
    "identity_relations",
    "is_one",
    "linearized_view",
    "matmul_relations",
    "readnone_relations",
    "state",
    "unread",
]
