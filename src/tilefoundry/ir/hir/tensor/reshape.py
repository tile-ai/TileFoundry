"""HIR ``Reshape`` Op: change the logical shape of a tensor."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, pattern, types

from .view_layout import derive_view_layout


@expr.register_op(dialect='tf', category='tensor')
class Reshape(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    new_shape = expr.ParamDef(kind="attribute", annotation=tuple)


@access.register_relation(Reshape)
def reshape_relations(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Where a result coordinate sits in the source it was renamed from."""
    out_shape = tuple(call.target.new_shape)
    return access.iterating(
        out_shape,
        (
            access.linearized_view(out_shape, tuple(ctx.type_of(call.args[0]).shape)),
            access.identity(len(out_shape)),
        ),
    )


@access.register_alias(Reshape)
def buffer_alias(call: expr.Call) -> int:
    return 0


def is_induction_var_singleton_reshape(node) -> bool:
    """Whether ``expr`` has the singleton view form for a loop index.

    HIR induction variables use ``UMAT`` storage; the exact induction-variable
    identity is checked by affine extraction, while this predicate keeps the
    preflight and partition gates aligned with that representation.
    """
    return (
        isinstance(node, expr.Call)
        and isinstance(node.target, Reshape)
        and node.args[0].type.shape == ()
        and node.args[0].type.storage is expr.StorageKind.UMAT
        and node.type.shape == (1,)
    )


def carry_sharded_reshape(layout: types.ShardLayout, new_shape: tuple):
    """Carry static sharding across an aligned view reshape.

    Layout positions may merge or divide at exact factor boundaries. A divided
    Split position must preserve a mesh-sized Split factor; singleton axes and
    mesh-only states carry freely. Unrepresentable alignment returns ``None``.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    axis_layout = layout.layout
    axis_shape = axis_layout.shape
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in axis_shape):
        return None
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in new_shape):
        return None

    axis_strides = axis_layout.strides
    n_axis = len(axis_shape)

    mesh_shape = types.flatten(layout.mesh.layout).shape
    split_mesh_extent: dict[int, int] = {}
    for mesh_axis_idx, attr in enumerate(layout.attrs):
        if isinstance(attr, types.Split) and mesh_axis_idx < len(mesh_shape):
            split_mesh_extent[attr.axis] = int(mesh_shape[mesh_axis_idx])

    def next_nonunit(start: int) -> int:
        i = start
        while i < n_axis and int(axis_shape[i]) == 1:
            i += 1
        return i

    new_positions: list[tuple[int, int, int | None]] = []
    old_to_new: dict[int, int] = {}
    ci = 0
    pending: tuple[int, int, int | None] | None = None
    for dim_value in new_shape:
        d = int(dim_value)
        if d == 1:
            new_positions.append((1, 0, None))
            continue
        prod = 1
        while prod < d:
            if pending is not None:
                cs, stride, old_pos = pending
                pending = None
            else:
                ci = next_nonunit(ci)
                if ci >= n_axis:
                    return None
                cs = int(axis_shape[ci])
                stride = axis_strides[ci]
                old_pos = ci
                ci += 1
            new_prod = prod * cs
            if new_prod <= d:
                if old_pos is not None:
                    old_to_new[old_pos] = len(new_positions)
                new_positions.append((cs, stride, old_pos))
                prod = new_prod
                continue

            if d % prod != 0:
                return None
            needed = d // prod
            if cs % needed != 0:
                return None
            residual = cs // needed
            mesh_ext = split_mesh_extent.get(old_pos) if old_pos is not None else None
            if mesh_ext is not None and needed % mesh_ext != 0:
                return None
            base_stride = stride * residual
            if mesh_ext is not None and needed != mesh_ext:
                outer_residual = needed // mesh_ext
                old_to_new[old_pos] = len(new_positions)
                new_positions.append((mesh_ext, base_stride * outer_residual, old_pos))
                new_positions.append((outer_residual, base_stride, None))
            else:
                if old_pos is not None:
                    old_to_new[old_pos] = len(new_positions)
                new_positions.append((needed, base_stride, old_pos))
            pending = (residual, stride, None)
            prod = d
    if pending is not None or next_nonunit(ci) < n_axis:
        return None

    new_attrs = []
    for attr in layout.attrs:
        if isinstance(attr, types.Split):
            if attr.axis not in old_to_new:
                return None
            new_attrs.append(replace(attr, axis=old_to_new[attr.axis]))
        else:
            new_attrs.append(attr)

    out_shape = tuple(s for s, _, _ in new_positions)
    out_strides = tuple(st for _, st, _ in new_positions)
    new_layout = types.Layout(shape=out_shape, strides=out_strides)
    return types.ShardLayout(layout=new_layout, attrs=tuple(new_attrs), mesh=layout.mesh)


@typeinfer.register(Reshape)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_shape = tuple(call.target.new_shape)

    genuine_sharding = isinstance(x_ty.layout, types.ShardLayout) and any(
        isinstance(attr, (types.Split, types.Partial)) for attr in x_ty.layout.attrs
    )
    if x_ty.storage is expr.StorageKind.UMAT and new_shape == () and not genuine_sharding:
        return types.TensorType.from_umat_scalar(x_ty.dtype)

    new_layout = None
    if isinstance(x_ty.layout, types.ShardLayout):
        new_layout = carry_sharded_reshape(x_ty.layout, new_shape)
        if new_layout is None and genuine_sharding:
            ctx.error(
                call,
                "Reshape cannot express the sharded layout: new shape does "
                "not align with the input layout factorization",
            )
        if new_layout is None:
            new_layout = replace(x_ty.layout, layout=types.Layout(new_shape))
    else:

        def reshaped(layout: types.Layout) -> types.Layout | None:
            nonunit = tuple(axis for axis, extent in enumerate(layout.shape) if extent != 1)
            if (
                len(nonunit) < len(layout.shape)
                and new_shape == tuple(layout.shape[axis] for axis in nonunit)
            ):
                return types.Layout(new_shape, tuple(layout.strides[axis] for axis in nonunit))
            expected = types.Layout(layout.shape).strides
            if layout.strides != expected:
                return None
            return types.Layout(new_shape)

        new_layout = derive_view_layout(x_ty, new_shape, reshaped)
    if new_layout is None:
        ctx.error(call, f"Reshape cannot preserve {type(x_ty.layout).__name__} layout")
    return types.TensorType(
        shape=new_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


def bound_or_inferred(dim_value, bindings):
    if any(name not in bindings for name in dim.var_names(dim_value)):
        return -1
    return evaluator.resolve_dim(dim_value, bindings)


@evaluator.register(Reshape)
def eval_reshape(ctx):
    """Reshape to the result type; each piece of a local view to its local type."""

    def shape_of(type):
        shape = tuple(bound_or_inferred(d, ctx.dim_bindings) for d in type.shape)
        if -1 in shape:
            shape = tuple(bound_or_inferred(d, ctx.dim_bindings) for d in ctx.op.new_shape)
        if shape.count(-1) > 1:
            raise evaluator.EvalError(
                "reshape: at most one dynamic axis can be inferred, "
                f"got new_shape={ctx.op.new_shape!r}"
            )
        return shape

    x = ctx.args[0].tensor
    if x.is_global:
        held = x.tensor.reshape(shape_of(ctx.result_type))
    else:
        shape = shape_of(core.local_type(ctx.result_type))
        held = tuple((p, t.reshape(shape)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(Reshape)
def reshape_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.Cost({}, cost.idle(call))


__all__ = ["Reshape"]
