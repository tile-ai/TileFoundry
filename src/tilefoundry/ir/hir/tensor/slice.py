from __future__ import annotations

import isl

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue
from tilefoundry.ir.core import Expr, Op, Tuple
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMul,
    DimSub,
    simplify_dim,
)
from tilefoundry.ir.types.dim_isl import dim_range
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard import (
    ComposedLayout,
    Layout,
    ShardLayout,
)
from tilefoundry.ir.types.shard.shard_layout import (
    layout_axis_to_tensor_axis,
    split_target_axes,
)
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AffineAccess,
    identity_access,
    register_access_relation,
    view_relations,
)


@register_op
class Slice(Op):
    x = ParamDef(kind="input", pattern=Tensor)
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
        reach = (
            name
            if self.size == "1"
            else f"{name} + ({self.size} - 1) * {self.stride}"
        )
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
    parameters: list[tuple[str, object]] = []
    for axis in range(len(logical_source.shape)):
        term = _Axis(
            axis,
            logical_source.shape[axis],
            sizes[axis] if axis < len(sizes) else 1,
            strides[axis] if axis < len(strides) else 1,
        )
        begin = term.start(offsets[axis] if axis < len(offsets) else 0, axis)
        parameters.extend(term.params)
        guards.extend(term.guards)
        walked = carried.get(axis, "0")
        stepped = walked if term.stride == "1" else f"{term.stride} * ({walked})"
        reads.append(stepped if begin == "0" else f"{stepped} + {begin}")

    names = [name for name, _value in parameters]
    prefix = f"[{', '.join(names)}] -> " if names else ""
    where = f" : {' and '.join(guards)}" if guards else ""
    rank = len(sizes) or len(logical_source.shape)
    domain = ", ".join(f"d{index}" for index in range(rank))
    return (
        AffineAccess(
            isl.map(f"{prefix}{{ [{domain}] -> [{', '.join(reads)}]{where} }}"),
            tuple(parameters),
        ),
        identity_access(rank),
    )


register_access_relation(Slice)(
    view_relations(
        0,
        _slice_view,
        over=lambda call, ctx: call.target.sizes,
    )
)


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


def _slice_shard_layout(call, ctx, x_ty, source, starts, inherited_offset):
    """Retain distribution when a slice leaves every split logical axis whole."""
    op = call.target
    static_starts = tuple(_constant_int(start) for start in starts.elements)
    narrow_axes = {
        axis
        for axis, (start, size, stride, extent) in enumerate(
            zip(static_starts, op.sizes, op.strides, x_ty.shape)
        )
        if start != 0 or size != extent or stride != 1
    }
    split_targets = split_target_axes(source, x_ty.shape)
    for mesh_axis, tensor_axis in enumerate(split_targets):
        if tensor_axis in narrow_axes:
            ctx.error(
                call,
                f"Slice narrows axis {tensor_axis}, which mesh axis {mesh_axis} "
                "splits; a window need not align with the mesh division. Slice "
                "before placing, or reshard to a layout that leaves axis "
                f"{tensor_axis} whole.",
            )

    base = source.layout
    if not isinstance(base, Layout):
        ctx.error(call, "Slice of a ShardLayout requires a primitive underlying Layout")
    position_to_axis = layout_axis_to_tensor_axis(base.shape, x_ty.shape)
    new_shape = list(base.shape)
    new_strides = None if base.strides is None else list(base.strides)
    narrow_positions: dict[int, int] = {}
    for tensor_axis in narrow_axes:
        positions = [
            position
            for position, mapped_axis in enumerate(position_to_axis)
            if mapped_axis == tensor_axis
        ]
        if len(positions) != 1:
            ctx.error(
                call,
                f"Slice narrows axis {tensor_axis}, whose layout uses positions "
                f"{positions}; the window cannot be represented by one layout axis",
            )
        position = positions[0]
        narrow_positions[tensor_axis] = position
        new_shape[position] = op.sizes[tensor_axis]
        if new_strides is not None:
            new_strides[position] = _dim_mul(
                new_strides[position], op.strides[tensor_axis]
            )

    sharded = ShardLayout(
        layout=Layout(
            shape=tuple(new_shape),
            strides=None if new_strides is None else tuple(new_strides),
        ),
        attrs=source.attrs,
        mesh=source.mesh,
    )
    if any(start is None for start in static_starts) or not all(
        isinstance(stride, int) and not isinstance(stride, bool)
        for stride in op.strides
    ):
        return sharded

    offset = inherited_offset
    if base.strides is None:
        return sharded
    for tensor_axis, start in enumerate(static_starts):
        if start == 0:
            continue
        position = narrow_positions[tensor_axis]
        source_stride = base.strides[position]
        if not isinstance(source_stride, int) or isinstance(source_stride, bool):
            return sharded
        offset += start * source_stride
    if offset == 0:
        return sharded
    return ComposedLayout(inner=None, offset=offset, outer=sharded)


@register_typeinfer(Slice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
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
    layout_shape = shape
    source = x_ty.layout
    inherited_offset = 0
    if (
        isinstance(source, ComposedLayout)
        and source.inner is None
        and isinstance(source.outer, (Layout, ShardLayout))
    ):
        inherited_offset = source.offset
        source = source.outer

    new_layout = None
    if isinstance(source, ShardLayout):
        new_layout = _slice_shard_layout(
            call, ctx, x_ty, source, starts, inherited_offset
        )
    elif isinstance(source, Layout) and source.strides is not None:
        static_starts = []
        steps = []
        for start, stride in zip(starts.elements, op.strides):
            if not (
                isinstance(start, Constant)
                and isinstance(start.value, int)
                and isinstance(stride, int)
                and not isinstance(stride, bool)
            ):
                break
            static_starts.append(int(start.value))
            steps.append(stride)
        else:
            new_layout = ComposedLayout(
                inner=None,
                offset=inherited_offset
                + sum(
                    start * stride
                    for start, stride in zip(static_starts, source.strides)
                ),
                outer=Layout(
                    shape=layout_shape,
                    strides=tuple(stride * step for stride, step in zip(source.strides, steps)),
                ),
            )
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
                f"Slice: invalid window on axis {axis}: start={start}, size={size}, "
                f"stride={stride}"
            )
        last = start if size == 0 else start + (size - 1) * stride
        if size and last >= ctx.args[0].data.shape[axis]:
            raise EvalError(
                f"Slice window exceeds axis {axis}: start={start}, size={size}, "
                f"stride={stride}, extent={ctx.args[0].data.shape[axis]}"
            )
        key.append(slice(start, start + size * stride, stride))
    return TensorValue(data=ctx.args[0].data[tuple(key)], type=ctx.result_type)


__all__ = ["Slice", "slice_size", "window_base"]
