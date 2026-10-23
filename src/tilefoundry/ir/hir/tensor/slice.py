from __future__ import annotations

from dataclasses import replace

import isl

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue
from tilefoundry.ir.core import Expr, Op, Tuple
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Swizzle, TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimSub, simplify_dim
from tilefoundry.ir.types.int_tuple import (
    axis_tiler,
    compatible,
    compatible_profile,
    flatten,
    mode_axes,
)
from tilefoundry.ir.types.layout import flatten as flat_layout
from tilefoundry.ir.types.layout import unflatten
from tilefoundry.ir.types.layout_algebra import composition, domain_offset
from tilefoundry.ir.types.shard_layout import Split
from tilefoundry.ir.types.utils import i64_const, static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    control_read,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias


@register_op(dialect='tf', category='tensor')
class Slice(Op):
    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    starts = ParamDef(kind="input")
    sizes = ParamDef(kind="attribute", annotation=tuple)
    strides = ParamDef(kind="attribute", annotation=tuple)

    def __init__(self, **attrs):
        super().__init__(**attrs)


def _slice_view(call: "Call", ctx, *, affine_starts=False) -> tuple:
    """Read stride*d + start; expand starts only for the mode-digit proof.

    Access relations retain the authored start as a parameter so lowering can
    reuse its expression. The window bounds also bound an opaque start.
    """
    rank = len(call.target.sizes)
    names = ", ".join(f"d{i}" for i in range(rank))
    values, reads = {}, None
    for axis, (begin, size, step, extent) in enumerate(zip(
        call.args[1].elements, call.target.sizes, call.target.strides,
        ctx.type_of(call.args[0]).shape, strict=True,
    )):
        coordinate = isl.pw_aff(f"{{ [{names}] -> [(d{axis})] }}")
        def affine(value):
            bound = dict(values)
            result = AccessRelation.affine_of(value, ctx, values=bound)
            values.update(bound)
            return result.insert_dims(isl.dim_type.IN, 0, rank)
        start = None
        if affine_starts or static_dim_value(begin) is not None:
            try:
                start = affine(begin)
            except (NotImplementedError, ArithmeticError, TypeError, ValueError, isl.Error):
                pass
        if start is None:
            name = AccessRelation.param_name(begin, ctx)
            values[name] = ctx.capture_root(begin)
            start = isl.pw_aff(f"[{name}] -> {{ [{names}] -> [({name})] }}")
        count, limit = affine(size), affine(extent)
        stride = static_dim_value(step)
        if stride is None or stride < 1:
            raise ValueError("Slice step must be a positive static integer")
        read = coordinate.scale(isl.val(stride)).add(start)
        valid = start.ge_set(affine(0)).intersect(
            start.add(count.sub(affine(1)).scale(isl.val(stride))).lt_set(limit)
        )
        read = read.intersect_domain(valid)
        part = isl.map.from_pw_aff(read)
        reads = part if reads is None else reads.flat_range_product(part)
    return AccessRelation(reads or isl.map("{ [] -> [] }"), values), identity_access(rank)


def _refuse_window(call, ctx, axis, source):
    mesh_axis = next((i for i, attr in enumerate(source.attrs) if isinstance(attr, Split)), 0) if isinstance(source, ShardLayout) else 0
    ctx.error(call, f"Slice narrows axis {axis}, which mesh axis {mesh_axis} splits; "
              "a window need not align with the mesh division. Slice before placing, "
              f"or reshard to a layout that leaves axis {axis} whole.")


def _require_affine_window(call, ctx, source, tiled, steps):
    """Prove each original mode digit advances without carrying across a block."""
    from tilefoundry.analysis.loop_domain import loop_domain  # noqa: PLC0415
    from tilefoundry.ir.isl_interop import layout_to_isl_map, shape_to_isl_set  # noqa: PLC0415

    logical = ctx.type_of(call.args[0]).shape
    axes = mode_axes(compatible_profile(source.shape, logical))
    split = {attr.axis for attr in source.attrs if isinstance(attr, Split)} if isinstance(source, ShardLayout) else set()
    checked = {mode for mode, axis in enumerate(axes) if axes.count(axis) > 1 or mode in split}
    if not checked:
        return
    read, _ = _slice_view(call, ctx, affine_starts=True)
    values = dict(read.values)
    read = read._result(read.relation.intersect_domain(shape_to_isl_set(call.target.sizes, values)), values)
    relation = layout_to_isl_map(logical, source, values, divided=lambda _: ())
    walked = read.relation.apply_range(relation).as_pw_multi_aff()
    rank = len(call.target.sizes)
    origin = read.relation
    for axis in range(rank):
        origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
    origin = origin.apply_range(relation).project_out(isl.dim_type.IN, 0, rank).as_pw_multi_aff()
    window = layout_to_isl_map(call.target.sizes, tiled, values, divided=lambda _: ()).as_pw_multi_aff()
    loop_constraints = None
    for mode, step in enumerate(steps):
        if mode not in checked:
            continue
        before = origin.get_at(mode).insert_dims(isl.dim_type.IN, 0, rank)
        delta = walked.get_at(mode).sub(before).sub(window.get_at(mode).scale(isl.val(step)))
        if not delta.non_zero_set().is_empty():
            names = {delta.get_dim_name(isl.dim_type.PARAM, i) for i in range(delta.dim(isl.dim_type.PARAM))}
            needed = [i for i, (var, *_bounds) in enumerate(ctx.loops)
                      if AccessRelation.param_name(var, ctx) in names]
            if needed:
                if loop_constraints is None:
                    loops = tuple(tuple(ctx.capture_root(x) for x in loop) for loop in ctx.loops[:max(needed) + 1])
                    try:
                        loop_constraints = loop_domain(loops, values).relation.params()
                    except (NotImplementedError, ArithmeticError, TypeError, ValueError, isl.Error):
                        _refuse_window(call, ctx, axes[mode], source)
                delta = delta.intersect_params(loop_constraints)
            if not delta.non_zero_set().is_empty():
                _refuse_window(call, ctx, axes[mode], source)


def _tiler(call, ctx, source):
    """Compose per-mode windows, preserving whole axes and written unit modes.

    Each tile stride is its step in the corresponding source mode. The digit
    proof uses tile extents independently of the physical address strides.
    """
    primitive = source.layout if isinstance(source, ShardLayout) else source
    logical = ctx.type_of(call.args[0]).shape
    profile = compatible_profile(primitive.shape, logical)
    grouped = unflatten(flat_layout(primitive), profile)
    sizes = tuple(None if compatible(n, size) and step == 1 else size
                  for n, size, step in zip(logical, call.target.sizes, call.target.strides, strict=True))
    splits = {attr.axis for attr in source.attrs if isinstance(attr, Split)} if isinstance(source, ShardLayout) else set()
    try:
        tiler = axis_tiler(profile, sizes, steps=call.target.strides, splits=splits)
    except ValueError:
        axis = next((i for i, size in enumerate(sizes) if size is not None), 0)
        _refuse_window(call, ctx, axis, source)
    window = composition(grouped, tiler, major="row")
    def steps_of(shape, tile):
        if tile is None:
            return (1,) * len(flatten(shape))
        if isinstance(tile, Layout):
            return tuple(flatten(tile.strides))
        return tuple(x for s, t in zip(shape, tile, strict=True) for x in steps_of(s, t))
    mode_steps = tuple(x for s, t in zip(profile, tiler, strict=True) for x in steps_of(s, t))
    tiled = Layout(window.shape)
    _require_affine_window(call, ctx, source, tiled, mode_steps)
    flat = flat_layout(window)
    axes = mode_axes(profile)
    kept = tuple(i for i, (before, after) in enumerate(zip(flatten(primitive.shape), flat.shape, strict=True))
                 if static_dim_value(before) == 1 or static_dim_value(after) != 1
                 or axes.count(axes[i]) == 1)
    if splits.difference(kept):
        mode = next(iter(splits.difference(kept)))
        _refuse_window(call, ctx, mode_axes(profile)[mode], source)
    positions = {old: new for new, old in enumerate(kept)}
    if len(kept) != len(flat.shape):
        axes = mode_axes(profile)
        def modes(values, axis):
            group = tuple(values[i] for i in kept if axes[i] == axis)
            return group[0] if len(group) == 1 else group
        window = Layout(tuple(modes(flat.shape, a) for a in range(len(logical))),
                        tuple(modes(flat.strides, a) for a in range(len(logical))))
    if isinstance(source, ShardLayout):
        window = flat_layout(window) if not any(isinstance(x, tuple) for x in primitive.shape) else window
        window = replace(source, layout=window, attrs=tuple(
            Split(positions[attr.axis]) if isinstance(attr, Split) else attr for attr in source.attrs
        ))
    return window


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
    static = tuple(static_dim_value(value) for value in (begin, end, stride))
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


@register_typeinfer(Slice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    """Carry the source mapping onto the window, with no new plain-view offset.

    A composed swizzle retains its inner mapping and accumulates static origins;
    runtime origins remain in the access relation and view pointer. Any other
    composed inner is refused rather than treating its outer strides as the
    complete address mapping.
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
    shard = x_ty.layout if isinstance(x_ty.layout, ShardLayout) else None
    source = shard.layout if shard is not None else x_ty.layout
    composed = source if isinstance(source, ComposedLayout) else None
    if composed is not None:
        if composed.inner is not None and not (isinstance(composed.inner, Swizzle) and isinstance(composed.outer, Layout)):
            ctx.error(call, f"Slice cannot narrow a composed layout whose inner is {type(composed.inner).__name__} over {type(composed.outer).__name__}: the window would have to be proven against that mapping, not against the outer strides")
        source = replace(shard, layout=composed.outer) if shard is not None else composed.outer
    else:
        source = x_ty.layout
    try:
        new_layout = _tiler(call, ctx, source)
        if composed is not None:
            _, moved = domain_offset(starts.elements, composed.outer, x_ty.shape)
            shifted = composed
            if static_dim_value(moved) is not None:
                shifted, _ = domain_offset(starts.elements, composed, x_ty.shape)
            new_layout = replace(new_layout, layout=replace(shifted, outer=new_layout.layout)) if shard is not None else replace(shifted, outer=new_layout)
    except ValueError as error:
        ctx.error(call, str(error))
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


__all__ = ["Slice", "slice_size"]
