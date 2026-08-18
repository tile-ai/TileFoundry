from __future__ import annotations

import math

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
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard import (
    ComposedLayout,
    Layout,
    ShardLayout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import (
    layout_axis_to_tensor_axis,
    split_target_axes,
)
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    OperandValue,
    StorageEffectClaim,
    StorageEffectKind,
    StorageSpan,
    WindowAccess,
    dense,
    factored_image,
    logical_coordinates,
    register_access_relation,
    same_placement,
    self_image,
    static_bytes,
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


def _static_window(call: "Call", source: TensorType) -> tuple[int, ...] | None:
    """The window's per-axis starts, when every one of them is compile-time.

    A stride other than one, a runtime start, or a source extent that is not an
    integer leaves the window undescribable here, so no address follows from it.
    """
    op = call.target
    starts = call.args[1]
    if not isinstance(starts, Tuple):
        return None
    if any(stride != 1 for stride in op.strides):
        return None
    if not all(isinstance(size, int) and not isinstance(size, bool) for size in op.sizes):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in source.shape):
        return None
    values = tuple(_constant_int(start) for start in starts.elements)
    if any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def _slice_storage(call: "Call", ctx) -> StorageEffectClaim | None:
    """A window addresses its source; the address follows for one unbroken run.

    That run needs the axes past the narrowed one whole and the ones before it
    single: any other window is several runs, and naming one offset for it would
    place bytes that are not there. Such a window still forwards -- it reads its
    source and nothing else -- it just cannot be a piece of a coverage proof.
    """
    source, result = ctx.type_of(call.args[0]), ctx.type_of(call)
    span = _window_span(call, ctx, source, result)
    if span is None:
        return StorageEffectClaim(StorageEffectKind.FORWARD, (0,))
    return StorageEffectClaim(StorageEffectKind.FORWARD, (0,), (span,))


def _window_span(call: "Call", ctx, source, result) -> "StorageSpan | None":
    """Where the window sits in its source, when that is one unbroken run."""
    if not same_placement(source, result) or not dense(source):
        return None
    starts = _static_window(call, source)
    if starts is None:
        return None
    sizes, extents = tuple(call.target.sizes), tuple(source.shape)
    narrowed = [axis for axis, size in enumerate(sizes) if size != extents[axis]]
    axis = max(narrowed) if narrowed else 0
    if any(size != 1 for size in sizes[:axis]):
        return None
    size = static_bytes(ctx.type_of(call))
    if not size:
        return None
    element = size // max(math.prod(sizes), 1)
    strides = try_c_order_strides(extents)
    if strides is None:
        return None
    offset = sum(start * stride for start, stride in zip(starts, strides)) * element
    return StorageSpan(0, offset, size)


def _slice_view(call: "Call", ctx) -> tuple:
    """A window reads its own extent, offset by where it starts.

    The offsets are per logical axis, so the result's coordinates are rebuilt
    per logical axis, shifted there, and only then spread over the source's
    positions -- a layout may give either side more positions than the Op has
    axes. A start that only arrives at run time is not an affine coordinate, so
    the pattern says so with a window rather than pretending to a map, and a
    window is then compared as a window.
    """
    result = ctx.local_type_of(call)
    logical_result = ctx.type_of(call)
    source = ctx.local_type_of(call.args[0])
    logical_source = ctx.type_of(call.args[0])
    starts = _static_window(call, logical_source)
    extents = tuple(result.shape)
    if starts is None:
        return (
            WindowAccess(
                tuple(
                    OperandValue(operand=1, element=axis) for axis in range(len(extents))
                ),
                extents,
            ),
            WindowAccess(tuple(0 for _ in extents), extents),
        )
    carried = logical_coordinates(result, logical_result)
    reads = []
    for axis in range(len(logical_source.shape)):
        walked = carried.get(axis, "0")
        start = starts[axis] if axis < len(starts) else 0
        reads.append(walked if not start else f"{walked} + {start}")
    domain = ", ".join(f"d{index}" for index in range(len(result.shape)))
    image = ", ".join(factored_image(reads, source, logical_source))
    return (
        isl.multi_aff(f"{{ [{domain}] -> [{image}] }}"),
        self_image(result, logical_result),
    )


register_access_relation(Slice)(view_relations(0, _slice_storage, _slice_view))


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
    def visit_Constant(self, expr: Constant) -> tuple[Expr | None, int]:
        value = _constant_int(expr)
        return (None, value) if value is not None else (expr, 0)

    def visit_Call(self, expr: Call) -> tuple[Expr | None, int]:
        if not isinstance(expr.target, (DimAdd, DimSub)):
            return expr, 0
        sign = 1 if isinstance(expr.target, DimAdd) else -1
        left, right = expr.args
        right_offset = _constant_int(right)
        if right_offset is not None:
            base, offset = self.visit(left)
            return base, offset + sign * right_offset
        left_offset = _constant_int(left)
        if left_offset is not None and sign == 1:
            base, offset = self.visit(right)
            return base, offset + left_offset
        return expr, 0

    def default_visit(self, expr) -> tuple[Expr | None, int]:
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
