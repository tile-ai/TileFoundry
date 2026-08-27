from __future__ import annotations

from dataclasses import replace

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import ComposedLayout, try_c_order_strides
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_access,
    linearized_view,
    register_access_relation,
    view_relations,
)


@register_op
class Reshape(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    new_shape = ParamDef(kind="attribute", annotation=tuple)




def _reshape_view(call: "Call", ctx) -> tuple:
    """Where a result coordinate sits in the source it was renamed from."""
    out_shape = tuple(call.target.new_shape)
    return (
        linearized_view(out_shape, tuple(ctx.type_of(call.args[0]).shape)),
        identity_access(len(out_shape)),
    )


register_access_relation(Reshape)(
    view_relations(
        0,
        _reshape_view,
        over=lambda call, ctx: call.target.new_shape,
    )
)


def is_induction_var_singleton_reshape(expr) -> bool:
    """Whether ``expr`` has the singleton view form for a loop index.

    HIR induction variables use ``UMAT`` storage; the exact induction-variable
    identity is checked by affine extraction, while this predicate keeps the
    preflight and partition gates aligned with that representation.
    """
    return (
        isinstance(expr, Call)
        and isinstance(expr.target, Reshape)
        and expr.args[0].type.shape == ()
        and expr.args[0].type.storage is StorageKind.UMAT
        and expr.type.shape == (1,)
    )


def _carry_sharded_reshape(layout: ShardLayout, new_shape: tuple):
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

    mesh_shape = layout.mesh.layout.shape
    split_mesh_extent: dict[int, int] = {}
    for mesh_axis_idx, attr in enumerate(layout.attrs):
        if isinstance(attr, Split) and mesh_axis_idx < len(mesh_shape):
            split_mesh_extent[attr.axis] = int(mesh_shape[mesh_axis_idx])

    def _next_nonunit(start: int) -> int:
        i = start
        while i < n_axis and int(axis_shape[i]) == 1:
            i += 1
        return i

    new_positions: list[tuple[int, int, int | None]] = []
    old_to_new: dict[int, int] = {}
    ci = 0
    pending: tuple[int, int, int | None] | None = None
    for dim in new_shape:
        d = int(dim)
        if d == 1:
            new_positions.append((1, 0, None))
            continue
        prod = 1
        while prod < d:
            if pending is not None:
                cs, stride, old_pos = pending
                pending = None
            else:
                ci = _next_nonunit(ci)
                if ci >= n_axis:
                    return None
                cs = int(axis_shape[ci])
                stride = axis_strides[ci] if axis_strides is not None else 0
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
    if pending is not None or _next_nonunit(ci) < n_axis:
        return None

    new_attrs = []
    for attr in layout.attrs:
        if isinstance(attr, Split):
            if attr.axis not in old_to_new:
                return None
            new_attrs.append(replace(attr, axis=old_to_new[attr.axis]))
        else:
            new_attrs.append(attr)

    out_shape = tuple(s for s, _, _ in new_positions)
    out_strides = None if axis_strides is None else tuple(st for _, st, _ in new_positions)
    new_layout = Layout(shape=out_shape, strides=out_strides)
    return ShardLayout(layout=new_layout, attrs=tuple(new_attrs), mesh=layout.mesh)


@register_typeinfer(Reshape)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_shape = tuple(call.target.new_shape)

    genuine_sharding = isinstance(x_ty.layout, ShardLayout) and any(
        not isinstance(attr, Broadcast) for attr in x_ty.layout.attrs
    )
    if x_ty.storage is StorageKind.UMAT and new_shape == () and not genuine_sharding:
        return TensorType.umat_scalar(x_ty.dtype)

    new_layout = None
    if isinstance(x_ty.layout, ShardLayout):
        genuine = any(not isinstance(a, Broadcast) for a in x_ty.layout.attrs)
        new_layout = _carry_sharded_reshape(x_ty.layout, new_shape)
        if new_layout is None and genuine:
            ctx.error(
                call,
                "Reshape cannot express the sharded layout: new shape does "
                "not align with the input layout factorization",
            )
    else:
        source = x_ty.layout
        if isinstance(source, Layout):
            source_strides = source.strides
            expected_strides = try_c_order_strides(source.shape)
            if source_strides is None or source_strides == expected_strides:
                new_layout = Layout(
                    shape=new_shape,
                    strides=try_c_order_strides(new_shape),
                )
        elif (
            isinstance(source, ComposedLayout)
            and isinstance(source.outer, Layout)
            and (
                source.outer.strides is None
                or source.outer.strides == try_c_order_strides(source.outer.shape)
            )
        ):
            new_layout = ComposedLayout(
                inner=source.inner,
                offset=source.offset,
                outer=Layout(
                    shape=new_shape,
                    strides=try_c_order_strides(new_shape),
                ),
            )
    return TensorType(
        shape=new_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(Reshape)
def _eval_reshape(ctx):

    shape = tuple(
        int(d) if isinstance(d, int) and not isinstance(d, bool) else -1 for d in ctx.op.new_shape
    )
    if shape.count(-1) > 1:
        raise EvalError(
            f"reshape: at most one dynamic axis can be inferred, got new_shape={ctx.op.new_shape!r}"
        )
    return TensorValue(data=ctx.args[0].data.reshape(shape), type=ctx.result_type)
