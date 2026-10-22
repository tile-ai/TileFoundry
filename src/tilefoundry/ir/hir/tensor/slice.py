from __future__ import annotations

from dataclasses import replace
from math import prod

import isl

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue
from tilefoundry.ir.core import Expr, Op, Tuple
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.isl_interop import dim_range
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Swizzle, TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMod, DimMul, DimSub, simplify_dim
from tilefoundry.ir.types.int_tuple import flatten, partition_modes
from tilefoundry.ir.types.layout_algebra import composition
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
from tilefoundry.ir.types.shard_layout import Split
from tilefoundry.ir.types.stride import compact_row_major, crd2idx
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.types.utils import i64_const, static_dim_value
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    control_read,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias


@register_op
class Slice(Op):
    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    starts = ParamDef(kind="input")
    sizes = ParamDef(kind="attribute", annotation=tuple)
    strides = ParamDef(kind="attribute", annotation=tuple)

    def __init__(self, **attrs):
        super().__init__(**attrs)


class _Unbounded(ValueError):
    """A relation would have a parameter nothing can bound."""


def _literal(value) -> int | None:
    """*value* as a number, when it is one."""
    resolved = _constant_int(value) if isinstance(value, Expr) else value
    if isinstance(resolved, int) and not isinstance(resolved, bool):
        return resolved
    return None


def _declared_range(value) -> "tuple[int, int] | None":
    """The range *value*'s own dimensions declare it to run over, inclusive.

    Asked of the one thing that already answers it. Taking every dimension at its
    smallest and then at its largest is not the interval of an expression over
    them -- `n - m + 10` is smallest where `n` is and `m` is not -- so the bounds
    come from the shared dimension arithmetic rather than from a second reading of
    it here.
    """
    if not dim_vars_by_name(value):
        return None
    try:
        low, high = dim_range(value)
    except (ArithmeticError, TypeError, ValueError):
        return None
    return (low, high - 1)


class _Axis:
    """One logical axis of a window, as terms of the relation being built.

    Every number the axis is built from is either a coefficient of the map or a
    parameter of it bound to the value it is. A parameter nothing can bound is
    refused: a relation with a hole in it is not bounded, and calling it bounded
    would let a consumer count coordinates nobody said were reached.
    """

    def __init__(self, axis: int, extent, size, stride) -> None:
        self.params: list[tuple[str, object]] = []
        self.guards: list[str] = []
        self.extent = self._term(extent, f"n{axis}", floor=0)
        self.size = self._term(size, f"z{axis}", floor=0)
        self.stride = self._coefficient(stride, f"t{axis}")

    def _coefficient(self, value, name: str) -> str:
        """A stride, which has to be a number rather than a parameter.

        A stride multiplies the coordinate walked, so one nobody has chosen would
        put a parameter against a variable, which no bound rescues. No authored
        program reaches this: the parser refuses a run-time step, and a step
        written as a dimension is a number by the time anything asks. It guards
        the invariant, so a stride that is somehow not a number says so rather
        than being rendered into a relation that cannot hold.
        """
        literal = _literal(value)
        if literal is None:
            raise _Unbounded(
                f"{name} is not a number, and a stride multiplies the coordinate "
                f"it steps, so this window's reach is not an affine relation"
            )
        if literal < 1:
            raise _Unbounded(f"{name} steps by {literal}, which walks nowhere")
        return str(literal)

    def _term(self, value, name: str, *, floor: int) -> str:
        literal = _literal(value)
        if literal is not None:
            return str(literal)
        declared = _declared_range(value)
        if declared is None:
            raise _Unbounded(
                f"{name} states no range, so nothing bounds it; a window's "
                f"relation cannot be bounded without one"
            )
        low, high = max(declared[0], floor), declared[1]
        if high < low:
            raise _Unbounded(f"{name} declares {declared}, which leaves it nothing to be")
        self.params.append((name, value))
        self.guards.append(f"{low} <= {name} <= {high}")
        return name

    def start(self, value, axis: int) -> str:
        """The axis's own start, as a coefficient or a parameter it constrains.

        A window lands inside what it reads: its last element sits at
        `start + (size - 1) * stride`, and that has to be a position the axis
        has. The constraint is written against whatever the extent, the size and
        the stride turned out to be, so one relation says it however many of
        them are only known later. It is not clamped: a window that fits nowhere
        has no legal start, and saying it starts at zero would be inventing one.
        """
        literal = _literal(value)
        name = str(literal) if literal is not None else f"s{axis}"
        if literal is None:
            self.params.append((name, value))
        self.guards.append(f"0 <= {name}")
        reach = name if self.size == "1" else f"{name} + ({self.size} - 1) * {self.stride}"
        self.guards.append(f"{reach} <= {self.extent} - 1")
        return name


def _slice_view(call: "Call", ctx) -> tuple:
    """A window reads its own extent, at the stride and offset it was given.

    The offsets are per logical axis, so the result's coordinates are rebuilt per
    logical axis, scaled by the stride, shifted by the start, and only then spread
    over the source's positions -- a layout may give either side more positions
    than the Op has axes. Whatever the Op only learns later is a parameter bound
    to the value it is, and the axis constrains them together: a window lands
    inside what it reads, whichever of its extent, size, stride and start turn out
    to be numbers.
    """
    logical_source = ctx.type_of(call.args[0])
    given = call.args[1]
    offsets = given.elements if isinstance(given, Tuple) else (given,)
    sizes, strides = tuple(call.target.sizes), tuple(call.target.strides)
    carried = {axis: f"d{axis}" for axis in range(len(logical_source.shape))}

    reads: list[str] = []
    guards: list[str] = []
    values: dict[str, object] = {}
    for axis in range(len(logical_source.shape)):
        term = _Axis(
            axis,
            logical_source.shape[axis],
            sizes[axis] if axis < len(sizes) else 1,
            strides[axis] if axis < len(strides) else 1,
        )
        begin = term.start(offsets[axis] if axis < len(offsets) else 0, axis)
        values.update(term.params)
        guards.extend(term.guards)
        walked = carried.get(axis, "0")
        stepped = walked if term.stride == "1" else f"{term.stride} * ({walked})"
        reads.append(stepped if begin == "0" else f"{stepped} + {begin}")

    prefix = f"[{', '.join(values)}] -> " if values else ""
    where = f" : {' and '.join(guards)}" if guards else ""
    rank = len(sizes) or len(logical_source.shape)
    domain = ", ".join(f"d{index}" for index in range(rank))
    return (
        AccessRelation(
            isl.map(f"{prefix}{{ [{domain}] -> [{', '.join(reads)}]{where} }}"),
            values,
        ),
        identity_access(rank),
    )


@register_access_relation(Slice)
def _slice_relations(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    reads, writes = _slice_view(call, ctx)
    return iterating(
        call.target.sizes,
        (reads, control_read(len(call.target.sizes), ctx, call.args[1]), writes),
    )


@register_buffer_alias(Slice)
def _buffer_alias(call: "Call") -> int:
    return 0


def _i64(value: int) -> Constant:
    return i64_const(value)


def slice_size(begin: Expr, end: Expr, stride: Expr) -> Expr:
    """Return ``ceil((end - begin) / stride)`` as a dimension expression."""
    static = tuple(_constant_int(value) for value in (begin, end, stride))
    if all(value is not None for value in static):
        start, stop, step = static
        if step > 0 and stop < start:
            return _i64(0)

    diff = simplify_dim(DimSub, (end, begin))

    bump = simplify_dim(
        DimAdd,
        (diff, simplify_dim(DimSub, (stride, _i64(1)))),
    )
    return simplify_dim(DimFloorDiv, (bump, stride))


def _constant_int(expr: Expr) -> int | None:
    if (
        isinstance(expr, Constant)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
    ):
        return int(expr.value)
    return None


class _WindowBaseVisitor(ExprVisitor[tuple[Expr | None, int]]):
    def visit_Constant(self, expr: Constant, ctx=None) -> tuple[Expr | None, int]:
        value = _constant_int(expr)
        return (None, value) if value is not None else (expr, 0)

    def visit_Call(self, expr: Call, ctx=None) -> tuple[Expr | None, int]:
        if not isinstance(expr.target, (DimAdd, DimSub)):
            return expr, 0
        sign = 1 if isinstance(expr.target, DimAdd) else -1
        left, right = expr.args
        right_offset = _constant_int(right)
        if right_offset is not None:
            base, offset = self.visit(left, ctx)
            return base, offset + sign * right_offset
        left_offset = _constant_int(left)
        if left_offset is not None and sign == 1:
            base, offset = self.visit(right, ctx)
            return base, offset + left_offset
        return expr, 0

    def default_visit(self, expr, ctx=None) -> tuple[Expr | None, int]:
        return expr, 0


def window_base(start: Expr) -> "tuple[Expr | None, int]":
    """Split a window start into the base it moves off and its constant offset.

    A start moved by a compile-time offset (``i + C``) reports ``(i, C)``, a
    plain start reports itself at offset ``0``, and an integer constant reports
    ``(None, value)``. A start whose offset is not compile-time has no constant
    to take out, so it reports itself unmoved and its reader decides whether it
    models anything.
    """
    value = _constant_int(start)
    if value is not None:
        return None, value
    return _WindowBaseVisitor().visit(start)


def _dim_mul(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return left * right
    return simplify_dim(DimMul, (left, right))


def _aligned(start, size, loops, aliases):
    """Prove block alignment over the authored loop domain, without rewriting Exprs."""
    from tilefoundry.ir.isl_interop import dim_to_isl_pw_aff  # noqa: PLC0415

    number = static_dim_value(start)
    if number is not None:
        return number % size == 0
    values = {}
    names = [f"i{index}" for index in range(len(loops))]
    coords = {id(loop[0]): name for loop, name in zip(loops, names, strict=True)}
    coords.update({alias: coords[root] for alias, root in aliases.items() if root in coords})
    domain = isl.set(f"{{ [{', '.join(names)}] }}")
    try:
        for index, (_var, begin, end, step) in enumerate(loops):
            begin = dim_to_isl_pw_aff(begin, values, coords=coords)
            end = dim_to_isl_pw_aff(end, values, coords=coords)
            step = static_dim_value(step)
            if step is None or step <= 0:
                return False
            iv = isl.pw_aff(f"{{ [{', '.join(names)}] -> [{names[index]}] }}")
            domain = domain.intersect(begin.le_set(iv)).intersect(iv.lt_set(end))
            domain = domain.intersect(iv.sub(begin).mod(isl.val(step)).zero_set())
        position = dim_to_isl_pw_aff(start, values, coords=coords)
        return position.mod(isl.val(size)).non_zero_set().intersect(domain).is_empty()
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return False


def window_layout(source, logical_shape, starts, sizes, strides, ctx):
    """Compose logical-axis windows while retaining each surviving shard mode."""
    shard = source if isinstance(source, ShardLayout) else None
    primitive = shard.layout if shard is not None else source
    if shard is not None and isinstance(primitive, ComposedLayout):
        window, moved = window_layout(
            replace(shard, layout=primitive.outer), logical_shape, starts, sizes, strides, ctx
        )
        if moved != 0:
            return ComposedLayout(primitive.inner, primitive.offset + moved, window), 0
        return replace(shard, layout=replace(primitive, outer=window.layout)), 0
    if not isinstance(primitive, Layout):
        raise ValueError(f"Slice cannot preserve {type(primitive).__name__} layout")
    correspondence = layout_correspondence(primitive.shape, logical_shape)
    if correspondence.groups is None:
        raise ValueError(f"layout {primitive!r} crosses logical axes of {logical_shape}")
    split = {attr.axis for attr in shard.attrs if isinstance(attr, Split)} if shard else set()
    steps = tuple(flatten(primitive.strides))
    grouped = any(isinstance(mode, tuple) for mode in primitive.shape)
    shape_modes, stride_modes, tilers, positions = [], [], [], {}
    offset, known_offset = 0, True

    for axis, group in enumerate(correspondence.groups):
        extents = tuple(correspondence.extents[position] for position in group)
        walks = tuple(steps[position] for position in group)
        shape_modes.append(correspondence.axis_mode(primitive.shape, axis))
        stride_modes.append(correspondence.axis_mode(primitive.strides, axis))
        start, size, step = starts[axis], sizes[axis], strides[axis]
        distributed = bool(split.intersection(group))
        kept = list(zip(group, extents, strict=True))
        if _literal(start) == 0 and size == logical_shape[axis] and step == 1:
            tilers.append(None)
        elif shard is not None and (distributed or len(group) > 1):
            mesh_axis = next(
                (
                    i
                    for i, attr in enumerate(shard.attrs)
                    if isinstance(attr, Split) and attr.axis in group
                ),
                0,
            )
            refusal = (
                f"Slice narrows axis {axis}, which mesh axis {mesh_axis} splits; "
                "a window need not align with the mesh division. Slice before placing, "
                f"or reshard to a layout that leaves axis {axis} whole."
            )
            remaining = _literal(size)
            if (
                step != 1
                or remaining is None
                or remaining <= 0
                or not _aligned(
                    start,
                    remaining,
                    ctx.loops,
                    {key: id(value) for key, value in ctx.loop_aliases.items()},
                )
            ):
                raise ValueError(refusal)
            widths = tuple(_literal(extent) for extent in extents)
            if any(width is None or width <= 0 for width in widths) or prod(widths) % remaining:
                raise ValueError(refusal)
            try:
                leading, trailing = partition_modes(
                    zip(group, widths, strict=True), prod(widths) // remaining, split
                )
            except ValueError as error:
                raise ValueError(refusal) from error
            if any(position in split and width != 1 for position, width in leading):
                raise ValueError(refusal)
            kept = sorted(
                (*((position, width) for position, width in leading if width == 1), *trailing)
            )
            tile_shape = tuple(width for _position, width in kept) or (1,)
            tilers.append(Layout(tile_shape, compact_row_major(tile_shape)))
        else:
            kept = [(group[0], size)]
            tilers.append(Layout((size,), (step,)))
        base = len(positions)
        positions.update((position, base + index) for index, (position, _width) in enumerate(kept))
        if _literal(start) == 0:
            continue
        if any(_literal(walk) is None for walk in walks) or (
            _literal(start) is None and not distributed
        ):
            known_offset = False
            continue
        widths = dict(kept)
        physical = tuple(
            0 if position in split or (distributed and widths.get(position) == extent) else walk
            for position, extent, walk in zip(group, extents, walks, strict=True)
        )
        if len(group) == 1:
            offset = simplify_dim(DimAdd, (offset, _dim_mul(start, physical[0])))
        elif _literal(start) is not None:
            digits = layout_correspondence(extents, (logical_shape[axis],)).digits(_literal(start))
            offset += crd2idx(digits, extents, physical)
        else:
            below = 1
            for index in reversed(range(len(extents))):
                digit = simplify_dim(DimFloorDiv, (start, below))
                if index:
                    digit = simplify_dim(DimMod, (digit, extents[index]))
                offset = simplify_dim(DimAdd, (offset, _dim_mul(digit, physical[index])))
                below = _dim_mul(below, extents[index])
    window = composition(
        Layout(tuple(shape_modes), tuple(stride_modes)), tuple(tilers), major="row"
    )
    if shard is not None and not grouped:
        window = Layout(tuple(flatten(window.shape)), tuple(flatten(window.strides)))
    if shard is not None:
        window = replace(
            shard,
            layout=window,
            attrs=tuple(
                Split(positions[attr.axis]) if isinstance(attr, Split) else attr
                for attr in shard.attrs
            ),
        )
    return window, (
        _literal(offset) if _literal(offset) is not None else offset
    ) if known_offset else 0


@register_typeinfer(Slice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    """The window's type, its source layout carried where the window keeps it.

    A ``Swizzle`` survives a window. It says where an element lives, not which
    element a coordinate names, so it leaves the domain -- and the access
    relation stated over it -- alone; the window narrows that domain and
    shifts the index by a constant the composition's ``offset`` already holds.
    Any other composed ``inner`` is refused by name: a window cannot be proven
    against a mapping this does not know, and reporting the outer strides as
    the whole layout would be wrong rather than partial.
    """
    x_ty = ctx.type_of(call.args[0])
    starts = call.args[1]
    op = call.target
    rank = len(x_ty.shape)
    if not isinstance(starts, Tuple) or len(starts.elements) != rank:
        ctx.error(call, f"Slice starts tuple length must match input rank {rank}")
    if not (len(op.sizes) == len(op.strides) == rank):
        ctx.error(call, f"Slice sizes/strides rank must match input rank {rank}")
    for axis, start in enumerate(starts.elements):
        start_ty = ctx.type_of(start)
        if start_ty.shape != () or start_ty.dtype.name not in ("i32", "i64"):
            ctx.error(call, f"Slice start for axis {axis} must be a rank-0 integer")
    for axis, size in enumerate(op.sizes):
        if isinstance(size, int) and (isinstance(size, bool) or size < 0):
            ctx.error(call, f"Slice size for axis {axis} must be non-negative")
    for axis, stride in enumerate(op.strides):
        if isinstance(stride, int) and (isinstance(stride, bool) or stride <= 0):
            ctx.error(call, f"Slice stride for axis {axis} must be positive")
    shape = tuple(op.sizes)
    source = x_ty.layout
    inherited_offset = 0
    inherited_inner = None
    if isinstance(source, ComposedLayout) and isinstance(source.outer, (Layout, ShardLayout)):
        if isinstance(source.inner, Swizzle) and isinstance(source.outer, Layout):
            inherited_inner = source.inner
        elif source.inner is not None:
            ctx.error(
                call,
                f"Slice cannot narrow a composed layout whose inner is "
                f"{type(source.inner).__name__} over "
                f"{type(source.outer).__name__}: the window would have to be "
                f"proven against that mapping, not against the outer strides",
            )
        inherited_offset = source.offset
        source = source.outer
    elif isinstance(source, ComposedLayout) and source.inner is not None:
        ctx.error(
            call,
            f"Slice cannot narrow a composed layout whose inner is "
            f"{type(source.inner).__name__}: the window would have to be proven "
            f"against that mapping, not against the outer strides",
        )

    try:
        new_layout, moved = window_layout(
            source, x_ty.shape, starts.elements, op.sizes, op.strides, ctx
        )
    except ValueError as error:
        ctx.error(call, str(error))
    offset = (
        inherited_offset + moved
        if isinstance(inherited_offset, int) and isinstance(moved, int)
        else simplify_dim(DimAdd, (inherited_offset, moved))
    )
    if isinstance(x_ty.layout, ComposedLayout):
        new_layout = ComposedLayout(inherited_inner, offset, new_layout)
    return TensorType(shape=shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@register_eval(Slice)
def _eval_slice(ctx):
    op = ctx.op
    starts = ctx.args[1]
    if not isinstance(starts, TupleValue):
        raise EvalError("Slice: starts must evaluate to a tuple")
    start_values = [int(value.data.reshape(-1)[0].item()) for value in starts.elements]
    sizes = [resolve_dim(size, ctx.dim_bindings) for size in op.sizes]
    strides = [resolve_dim(stride, ctx.dim_bindings) for stride in op.strides]
    if not (len(start_values) == len(sizes) == len(strides) == ctx.args[0].data.ndim):
        raise EvalError("Slice: starts/sizes/strides rank must match input rank")
    key = []
    for axis, (start, size, stride) in enumerate(zip(start_values, sizes, strides)):
        if start < 0 or size < 0 or stride <= 0:
            raise EvalError(
                f"Slice: invalid window on axis {axis}: start={start}, size={size}, stride={stride}"
            )
        last = start if size == 0 else start + (size - 1) * stride
        if size and last >= ctx.args[0].data.shape[axis]:
            raise EvalError(
                f"Slice window exceeds axis {axis}: start={start}, size={size}, "
                f"stride={stride}, extent={ctx.args[0].data.shape[axis]}"
            )
        key.append(slice(start, start + size * stride, stride))
    return TensorValue(data=ctx.args[0].data[tuple(key)], type=ctx.result_type)


__all__ = ["Slice", "slice_size", "window_base", "window_layout"]
