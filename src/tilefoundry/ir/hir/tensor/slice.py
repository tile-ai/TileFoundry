"""HIR strided window ``Slice`` of a tensor."""

from __future__ import annotations

from dataclasses import replace

import isl

from tilefoundry import core, services
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, pattern, types, utils
from tilefoundry.ir.types import int_tuple, layout_algebra


@expr.register_op(dialect='tf', category='tensor')
class Slice(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    starts = expr.ParamDef(kind="input")
    sizes = expr.ParamDef(kind="attribute", annotation=tuple)
    strides = expr.ParamDef(kind="attribute", annotation=tuple)

    def __init__(self, **attrs):
        super().__init__(**attrs)


def slice_view(call: "expr.Call", ctx, *, affine_starts=False) -> tuple:
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
            result = access.affine_of(value, ctx, values=bound)
            values.update(bound)
            return result.insert_dims(isl.dim_type.IN, 0, rank)
        start = None
        if affine_starts or expr.static_dim_value(begin) is not None:
            try:
                start = affine(begin)
            except (NotImplementedError, ArithmeticError, TypeError, ValueError, isl.Error):
                pass
        if start is None:
            name = access.param_name(begin, ctx)
            values[name] = ctx.capture_root(begin)
            start = isl.pw_aff(f"[{name}] -> {{ [{names}] -> [({name})] }}")
        count, limit = affine(size), affine(extent)
        stride = expr.static_dim_value(step)
        if stride is None or stride < 1:
            raise ValueError("Slice step must be a positive static integer")
        read = coordinate.scale(isl.val(stride)).add(start)
        valid = start.ge_set(affine(0)).intersect(
            start.add(count.sub(affine(1)).scale(isl.val(stride))).lt_set(limit)
        )
        read = read.intersect_domain(valid)
        part = isl.map.from_pw_aff(read)
        reads = part if reads is None else reads.flat_range_product(part)
    return access.Relation(reads or isl.map("{ [] -> [] }"), values), access.identity(rank)


def refuse_window(call, ctx, axis, source):
    mesh_axis = next((i for i, attr in enumerate(source.attrs) if isinstance(attr, types.Split)), 0) if isinstance(source, types.ShardLayout) else 0
    ctx.error(call, f"Slice narrows axis {axis}, which mesh axis {mesh_axis} splits; "
              "a window need not align with the mesh division. Slice before placing, "
              f"or reshard to a layout that leaves axis {axis} whole.")


def require_affine_window(call, ctx, source, tiled, steps):
    """Prove each original mode digit advances without carrying across a block."""
    logical = ctx.type_of(call.args[0]).shape
    axes = int_tuple.mode_axes(int_tuple.compatible_profile(source.shape, logical))
    split = {attr.axis for attr in source.attrs if isinstance(attr, types.Split)} if isinstance(source, types.ShardLayout) else set()
    checked = {mode for mode, axis in enumerate(axes) if axes.count(axis) > 1 or mode in split}
    if not checked:
        return
    read, _ = slice_view(call, ctx, affine_starts=True)
    values = dict(read.values)
    read = read.aligned(read.relation.intersect_domain(dim.shape_to_isl_set(call.target.sizes, values)), values)
    relation = dim.layout_to_isl_map(logical, source, values, divided=lambda _: ())
    walked = read.relation.apply_range(relation).as_pw_multi_aff()
    rank = len(call.target.sizes)
    origin = read.relation
    for axis in range(rank):
        origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
    origin = origin.apply_range(relation).project_out(isl.dim_type.IN, 0, rank).as_pw_multi_aff()
    window = dim.layout_to_isl_map(call.target.sizes, tiled, values, divided=lambda _: ()).as_pw_multi_aff()
    loop_constraints = None
    for mode, step in enumerate(steps):
        if mode not in checked:
            continue
        before = origin.get_at(mode).insert_dims(isl.dim_type.IN, 0, rank)
        delta = walked.get_at(mode).sub(before).sub(window.get_at(mode).scale(isl.val(step)))
        if not delta.non_zero_set().is_empty():
            names = {delta.get_dim_name(isl.dim_type.PARAM, i) for i in range(delta.dim(isl.dim_type.PARAM))}
            needed = [i for i, (var, *bounds) in enumerate(ctx.loops)
                      if access.param_name(var, ctx) in names]
            if needed:
                if loop_constraints is None:
                    loops = tuple(tuple(ctx.capture_root(x) for x in loop) for loop in ctx.loops[:max(needed) + 1])
                    try:
                        loop_constraints = access.loop_domain(loops, values).relation.params()
                    except (NotImplementedError, ArithmeticError, TypeError, ValueError, isl.Error):
                        refuse_window(call, ctx, axes[mode], source)
                delta = delta.intersect_params(loop_constraints)
            if not delta.non_zero_set().is_empty():
                refuse_window(call, ctx, axes[mode], source)


def steps_of(shape, tile):
    """The step each flattened source mode takes inside its tile."""
    if tile is None:
        return (1,) * len(types.flatten(shape))
    if isinstance(tile, types.Layout):
        return tuple(types.flatten(tile.strides))
    return tuple(x for s, t in zip(shape, tile, strict=True) for x in steps_of(s, t))


def window_tiler(call, ctx, source):
    """Compose per-mode windows, preserving whole axes and written unit modes.

    Each tile stride is its step in the corresponding source mode. The digit
    proof uses tile extents independently of the physical address strides.
    """
    primitive = source.layout if isinstance(source, types.ShardLayout) else source
    logical = ctx.type_of(call.args[0]).shape
    profile = int_tuple.compatible_profile(primitive.shape, logical)
    grouped = types.unflatten(types.flatten(primitive), profile)
    sizes = tuple(None if int_tuple.compatible(n, size) and step == 1 else size
                  for n, size, step in zip(logical, call.target.sizes, call.target.strides, strict=True))
    splits = {attr.axis for attr in source.attrs if isinstance(attr, types.Split)} if isinstance(source, types.ShardLayout) else set()
    try:
        tiler = utils.axis_tiler(profile, sizes, steps=call.target.strides, splits=splits)
    except ValueError:
        axis = next((i for i, size in enumerate(sizes) if size is not None), 0)
        refuse_window(call, ctx, axis, source)
    window = layout_algebra.composition(grouped, tiler, major="row")
    mode_steps = tuple(x for s, t in zip(profile, tiler, strict=True) for x in steps_of(s, t))
    tiled = types.Layout(window.shape)
    require_affine_window(call, ctx, source, tiled, mode_steps)
    flat = types.flatten(window)
    axes = int_tuple.mode_axes(profile)
    kept = tuple(i for i, (before, after) in enumerate(zip(types.flatten(primitive.shape), flat.shape, strict=True))
                 if expr.static_dim_value(before) == 1 or expr.static_dim_value(after) != 1
                 or axes.count(axes[i]) == 1)
    if splits.difference(kept):
        mode = next(iter(splits.difference(kept)))
        refuse_window(call, ctx, int_tuple.mode_axes(profile)[mode], source)
    positions = {old: new for new, old in enumerate(kept)}
    if len(kept) != len(flat.shape):
        axes = int_tuple.mode_axes(profile)
        def modes(values, axis):
            group = tuple(values[i] for i in kept if axes[i] == axis)
            return group[0] if len(group) == 1 else group
        window = types.Layout(tuple(modes(flat.shape, a) for a in range(len(logical))),
                        tuple(modes(flat.strides, a) for a in range(len(logical))))
    if isinstance(source, types.ShardLayout):
        window = types.flatten(window) if not any(isinstance(x, tuple) for x in primitive.shape) else window
        window = replace(source, layout=window, attrs=tuple(
            types.Split(positions[attr.axis]) if isinstance(attr, types.Split) else attr for attr in source.attrs
        ))
    return window


@access.register_relation(Slice)
def slice_relations(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    reads, writes = slice_view(call, ctx)
    return access.iterating(
        call.target.sizes,
        (reads, access.control_read(len(call.target.sizes), ctx, call.args[1]), writes),
    )


@access.register_alias(Slice)
def buffer_alias(call: "expr.Call") -> int:
    return 0


def i64(value: int) -> expr.Constant:
    return dim.i64_const(value)


def slice_size(begin: expr.Expr, end: expr.Expr, stride: expr.Expr) -> expr.Expr:
    """Return ``ceil((end - begin) / stride)`` as a dimension expression."""
    static = tuple(expr.static_dim_value(value) for value in (begin, end, stride))
    if all(value is not None for value in static):
        start, stop, step = static
        if step > 0 and stop < start:
            return i64(0)

    diff = services.simplify_dim("sub", (end, begin))

    bump = services.simplify_dim(
        "add",
        (diff, services.simplify_dim("sub", (stride, i64(1)))),
    )
    return services.simplify_dim("floordiv", (bump, stride))


@typeinfer.register(Slice)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
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
    if not isinstance(starts, expr.Tuple) or len(starts.elements) != rank:
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
    shard = x_ty.layout if isinstance(x_ty.layout, types.ShardLayout) else None
    source = shard.layout if shard is not None else x_ty.layout
    composed = source if isinstance(source, types.ComposedLayout) else None
    if composed is not None:
        if composed.inner is not None and not (isinstance(composed.inner, types.Swizzle) and isinstance(composed.outer, types.Layout)):
            ctx.error(call, f"Slice cannot narrow a composed layout whose inner is {type(composed.inner).__name__} over {type(composed.outer).__name__}: the window would have to be proven against that mapping, not against the outer strides")
        source = replace(shard, layout=composed.outer) if shard is not None else composed.outer
    else:
        source = x_ty.layout
    try:
        new_layout = window_tiler(call, ctx, source)
        if composed is not None:
            _, moved = types.domain_offset(starts.elements, composed.outer, x_ty.shape)
            shifted = composed
            if expr.static_dim_value(moved) is not None:
                shifted, _ = types.domain_offset(starts.elements, composed, x_ty.shape)
            new_layout = replace(new_layout, layout=replace(shifted, outer=new_layout.layout)) if shard is not None else replace(shifted, outer=new_layout)
    except ValueError as error:
        ctx.error(call, str(error))
    return types.TensorType(shape=shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@evaluator.register(Slice)
def eval_slice(ctx):
    op = ctx.op
    starts = ctx.args[1]
    if not isinstance(starts, evaluator.TupleValue):
        raise evaluator.EvalError("Slice: starts must evaluate to a tuple")
    sizes = [evaluator.resolve_dim(size, ctx.dim_bindings) for size in op.sizes]
    strides = [evaluator.resolve_dim(stride, ctx.dim_bindings) for stride in op.strides]

    def slice_(x, *starts):
        start_values = [int(value.reshape(-1)[0].item()) for value in starts]
        if not (len(start_values) == len(sizes) == len(strides) == x.ndim):
            raise evaluator.EvalError("Slice: starts/sizes/strides rank must match input rank")
        key = []
        for axis, (start, size, stride) in enumerate(zip(start_values, sizes, strides)):
            if start < 0 or size < 0 or stride <= 0:
                raise evaluator.EvalError(
                    f"Slice: invalid window on axis {axis}: start={start}, size={size}, "
                    f"stride={stride}"
                )
            last = start if size == 0 else start + (size - 1) * stride
            if size and last >= x.shape[axis]:
                raise evaluator.EvalError(
                    f"Slice window exceeds axis {axis}: start={start}, size={size}, "
                    f"stride={stride}, extent={x.shape[axis]}"
                )
            key.append(slice(start, start + size * stride, stride))
        return x[tuple(key)]

    xs = tuple(arg.tensor for arg in (ctx.args[0], *starts.elements))
    placements = evaluator.common_placements(*xs)
    held = (
        slice_(*(x.tensor for x in xs))
        if not placements
        else tuple((p, slice_(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Slice", "slice_size"]


@cost.register(Slice)
def slice_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """A slice is a view; reading its coordinates still has a cost."""
    _, starts = cost.input_types(call, ctx)
    return cost.Cost(
        {},
        (
            cost.TrafficBytes(),
            cost.TrafficBytes(read=utils.tensor_bytes(starts)),
            cost.TrafficBytes(),
        ),
    )
