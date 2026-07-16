from __future__ import annotations

from dataclasses import replace

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
)


@register_op
class Reshape(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    new_shape = ParamDef(kind="attribute", annotation=tuple)


def _carry_sharded_reshape(layout: ShardLayout, new_shape: tuple):
    """Carry a genuine sharding across a reshape (a view) when the layout
    factorization aligns with *new_shape*.

    A reshape is a view: it inserts/removes size-1 axes and groups along
    boundaries. It can carry the sharding when every layout position lies
    entirely within one new axis (whole positions merge into a coarser new
    axis, in either order). It can ALSO carry the sharding when a layout
    position must itself divide across a new-axis boundary, provided that
    position is `Split`-bound and its outer (earlier) sub-factor is evenly
    divisible by the bound mesh extent: the sub-factor itself further
    factors into `(mesh_ext, sub_factor // mesh_ext)` — `Split` relocates to
    the `mesh_ext`-sized position (local extent 1, per
    `docs/spec/shard.md` §7.1.1), and both the `sub_factor // mesh_ext`
    remainder and the inner residual carry forward as plain (non-`Split`)
    layout positions. A plain (non-`Split`) position may divide at any
    boundary that evenly factors it; only a `Split`-bound position whose
    boundary the mesh extent does not divide fails closed. Size-1 axes are
    inserted/dropped freely and hold no sharding. `Partial` / `Broadcast`
    carry through unchanged (mesh-axis states, no layout axis).

    Returns the carried ``ShardLayout``, or ``None`` when the reshape cannot
    express the sharding (the caller fails closed). All extents must be static
    to verify alignment.
    """
    axis_layout = layout.layout
    axis_shape = axis_layout.shape
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in axis_shape):
        return None
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in new_shape):
        return None

    axis_strides = axis_layout.strides
    n_axis = len(axis_shape)

    # The mesh extent bound to each Split-carrying layout axis (at most one
    # Split per axis -- ShardLayout construction forbids two Splits sharing
    # an axis).
    mesh_shape = layout.mesh.shape
    split_mesh_extent: dict[int, int] = {}
    for mesh_axis_idx, attr in enumerate(layout.attrs):
        if isinstance(attr, Split) and mesh_axis_idx < len(mesh_shape):
            split_mesh_extent[attr.axis] = int(mesh_shape[mesh_axis_idx])

    def _next_nonunit(start: int) -> int:
        i = start
        while i < n_axis and int(axis_shape[i]) == 1:
            i += 1
        return i

    # Walk new axes; compose each non-size-1 new axis from a contiguous run of
    # layout positions, recording the old layout position -> new layout position
    # remap. A position that overshoots the current new axis divides into an
    # outer sub-factor (completing the axis) and an inner residual carried
    # forward via `pending` to the next axis; only the outer sub-factor keeps
    # the position's `old_pos` (so only it is eligible for a `Split` remap) --
    # the residual is a fresh, unsharded position regardless of what it
    # descends from. Inserted size-1 new axes get a fresh unit position.
    new_positions: list[tuple[int, int, int | None]] = []  # (size, stride, old_pos)
    old_to_new: dict[int, int] = {}
    ci = 0
    pending: tuple[int, int, int | None] | None = None  # (size, stride, old_pos)
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
                    return None  # ran out of layout positions to compose this axis
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
            # This position overshoots the axis: split into an outer
            # sub-factor (`needed`, completes the axis) and an inner
            # residual (carried to the next axis via `pending`).
            if d % prod != 0:
                return None  # axis boundary does not land on a position boundary
            needed = d // prod
            if cs % needed != 0:
                return None  # position does not divide at the axis boundary
            residual = cs // needed
            mesh_ext = split_mesh_extent.get(old_pos) if old_pos is not None else None
            if mesh_ext is not None and needed % mesh_ext != 0:
                return None  # Split-bound: mesh extent must divide the outer sub-factor
            base_stride = stride * residual
            if mesh_ext is not None and needed != mesh_ext:
                # The outer sub-factor exceeds the mesh extent: factor it
                # further into (mesh_ext, Split-bound, local extent 1) and
                # (needed // mesh_ext, plain), mirroring
                # parser/sugar.py::_canonicalize_single_axis, so the
                # Split-bound layout dim keeps local_shape == 1.
                outer_residual = needed // mesh_ext
                old_to_new[old_pos] = len(new_positions)
                new_positions.append(
                    (mesh_ext, base_stride * outer_residual, old_pos)
                )
                new_positions.append((outer_residual, base_stride, None))
            else:
                if old_pos is not None:
                    old_to_new[old_pos] = len(new_positions)
                new_positions.append((needed, base_stride, old_pos))
            pending = (residual, stride, None)
            prod = d
    if pending is not None or _next_nonunit(ci) < n_axis:
        return None  # leftover layout content cannot be placed

    new_attrs = []
    for attr in layout.attrs:
        if isinstance(attr, Split):
            if attr.axis not in old_to_new:
                return None  # a sharded layout position was dropped -> fail closed
            new_attrs.append(replace(attr, axis=old_to_new[attr.axis]))
        else:
            # Partial / Broadcast are mesh-axis states with no layout axis; they
            # carry through the reshape unchanged.
            new_attrs.append(attr)

    out_shape = tuple(s for s, _, _ in new_positions)
    out_strides = (
        None if axis_strides is None else tuple(st for _, st, _ in new_positions)
    )
    new_layout = Layout(shape=out_shape, strides=out_strides)
    return ShardLayout(layout=new_layout, attrs=tuple(new_attrs), mesh=layout.mesh)


@register_typeinfer(Reshape)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_shape = tuple(call.target.new_shape)

    new_layout = None
    if isinstance(x_ty.layout, ShardLayout):
        genuine = any(not isinstance(a, Broadcast) for a in x_ty.layout.attrs)
        if not genuine:
            new_layout = None  # replicated input -> unsharded output
        else:
            new_layout = _carry_sharded_reshape(x_ty.layout, new_shape)
            if new_layout is None:
                # A genuine sharding whose layout factorization does not align with
                # the new shape cannot be expressed; fail closed rather than
                # fabricate a layout. (Re-laying-out across a misaligned reshape
                # would need an explicit Reshard.)
                ctx.error(
                    call,
                    "Reshape cannot express the sharded layout: new shape does "
                    "not align with the input layout factorization",
                )
    return TensorType(
        shape=new_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(Reshape)
def _eval_reshape(ctx):
    # A symbolic (DimVar / Expr) target axis is inferred from the concrete input
    # via torch's ``-1``; at most one axis may be inferred.
    shape = tuple(
        int(d) if isinstance(d, int) and not isinstance(d, bool) else -1
        for d in ctx.op.new_shape
    )
    if shape.count(-1) > 1:
        raise EvalError(
            "reshape: at most one dynamic axis can be inferred, "
            f"got new_shape={ctx.op.new_shape!r}"
        )
    return TensorValue(data=ctx.args[0].data.reshape(shape), type=ctx.result_type)
