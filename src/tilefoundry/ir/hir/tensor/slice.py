from __future__ import annotations

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
)
from tilefoundry.ir.types.shard.shard_layout import (
    layout_axis_to_tensor_axis,
    split_target_axes,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Slice(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    starts = ParamDef(kind="input")
    sizes = ParamDef(kind="attribute", annotation=tuple)
    strides = ParamDef(kind="attribute", annotation=tuple)

    def __init__(self, **attrs):
        super().__init__(**attrs)


register_access_relation(Slice)(identity_relations(2))


def _i64(value: int) -> Constant:
    return i64_const(value)


def slice_size(begin: Expr, end: Expr, stride: Expr) -> Expr:
    """Return ``max(0, ceil((end - begin) / stride))`` as a dimension Expr.

    Constant chains fold through ``simplify_dim``. A non-positive constant
    stride returns zero explicitly because generic arithmetic folding does not
    encode slice-domain semantics.
    """
    if isinstance(begin, Constant) and isinstance(end, Constant) and isinstance(stride, Constant):
        b, e, s = int(begin.value), int(end.value), int(stride.value)
        if s <= 0:
            return _i64(0)
        n = max(0, (e - b + s - 1) // s)
        return _i64(n)
    if isinstance(end, Call) and isinstance(end.target, DimAdd) and end.args[0] is begin:
        window = end.args[1]
        if isinstance(stride, Constant) and stride.value == 1:
            return window
        bump = simplify_dim(
            DimAdd,
            (window, simplify_dim(DimSub, (stride, _i64(1)))),
        )
        return simplify_dim(DimFloorDiv, (bump, stride))
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


__all__ = ["Slice", "slice_size"]
