"""Enforce shared HIR shard-soundness rules during type inference.

Checks cover elementwise commutation, multilinear combinations, and matching
Partial state for in-place-style operations.

See [shard §8](docs/spec/shard.md#8-layout-propagation).
"""

from __future__ import annotations

from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    ShardLayout,
    shard_layout_of,
)
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


def reject_partials(ctx, call, arg_name, layout, commutes_with=frozenset()):
    """Elementwise Partial-commutation check on a single operand.

    Every mesh-axis Partial reduction on *layout* must be in *commutes_with*
    (default: none commute, so any Partial state is rejected).
    """
    for axis, reduction in enumerate(partial_reductions_by_axis(layout)):
        if reduction is None or reduction in commutes_with:
            continue
        ctx.error(
            call,
            f"{arg_name} carries Partial({reduction}) on mesh axis {axis}, "
            f"which does not commute; insert reshard({arg_name}, Broadcast) "
            "before this consumer",
        )


def _is_replicated_at(layout, axis: int) -> bool:
    layout = shard_layout_of(layout)
    if layout is None or axis >= len(layout.attrs):
        return True
    return isinstance(layout.attrs[axis], Broadcast)


def check_multilinear_partials(
    ctx,
    call,
    named_operands,
    allowed_reduction="sum",
    *,
    anchor=None,
    commutes_jointly=frozenset(),
):
    """Check Partial commutation across multilinear operands per mesh axis.

    Normally one permitted Partial may combine with replicated peers. Multiple
    operands may participate only for a shared ``commutes_jointly`` reduction;
    ``anchor`` restricts which operand's state the output preserves.
    """
    allowed = {allowed_reduction} if isinstance(allowed_reduction, str) else set(allowed_reduction)
    states = {name: partial_reductions_by_axis(ty.layout) for name, ty in named_operands}
    axis_count = max((len(s) for s in states.values()), default=0)
    for axis in range(axis_count):
        partials = [
            (name, states[name][axis])
            for name, _ in named_operands
            if axis < len(states[name]) and states[name][axis] is not None
        ]
        if not partials:
            continue
        if anchor is not None:
            for name, reduction in partials:
                if name != anchor:
                    ctx.error(
                        call,
                        f"{name} carries Partial({reduction}) on mesh axis {axis}; "
                        "the output cannot preserve this secondary state. Use "
                        f"reshard({name}, Broadcast) before this consumer",
                    )
        if len(partials) > 1:
            reductions_here = {reduction for _, reduction in partials}
            if len(reductions_here) == 1 and next(iter(reductions_here)) in commutes_jointly:
                continue
            details = ", ".join(f"{name}=Partial({reduction})" for name, reduction in partials)
            ctx.error(
                call,
                f"multiple value-carrying Partials on mesh axis {axis} "
                f"({details}) do not commute here; reshard to Broadcast "
                "before this consumer",
            )
        name, reduction = partials[0]
        if reduction not in allowed:
            allowed_text = " or ".join(sorted(allowed)) if allowed else "no reduction"
            ctx.error(
                call,
                f"{name} carries Partial({reduction}) on mesh axis {axis}; "
                f"commutes with {allowed_text} only. Insert reshard({name}, "
                "Broadcast) before this consumer",
            )
        for other_name, other_ty in named_operands:
            if other_name == name:
                continue
            if not _is_replicated_at(other_ty.layout, axis):
                ctx.error(
                    call,
                    f"{name} carries Partial({reduction}) on mesh axis {axis}, but "
                    f"{other_name} is not Broadcast/replicated on that axis. "
                    f"Reshard {other_name} to Broadcast before this consumer",
                )


def require_matching_partial_state(ctx, call, dst, update, dst_name, update_name):
    """Require identical per-mesh-axis Partial state for an in-place-style write."""
    dst_partials = [
        (axis, reduction)
        for axis, reduction in enumerate(partial_reductions_by_axis(dst.layout))
        if reduction is not None
    ]
    update_partials = [
        (axis, reduction)
        for axis, reduction in enumerate(partial_reductions_by_axis(update.layout))
        if reduction is not None
    ]
    dst_layout = shard_layout_of(dst.layout)
    update_layout = shard_layout_of(update.layout)
    if dst_partials:
        if not (
            dst_layout is not None
            and update_layout is not None
            and update_layout.mesh == dst_layout.mesh
            and update_layout.attrs == dst_layout.attrs
        ):
            axis, reduction = dst_partials[0]
            ctx.error(
                call,
                f"{dst_name} carries a Partial({reduction}) on mesh axis {axis}; "
                f"{update_name} must carry the identical per-mesh-axis state. "
                f"Insert Reshard({update_name}, Broadcast) or match {dst_name} "
                "before this consumer",
            )
    elif update_partials:
        axis, reduction = update_partials[0]
        ctx.error(
            call,
            f"{update_name} carries Partial({reduction}) on mesh axis {axis}, "
            f"but {dst_name} is complete; insert reshard({update_name}, "
            "Broadcast) before this consumer",
        )


def reject_dynamic_shards(ctx, call, types, op_name: str) -> None:
    """Reject ownership that cannot be propagated statically."""
    for index, type_ in enumerate(types):
        layout = shard_layout_of(type_.layout)
        if layout is not None and any(isinstance(attr, Dynamic) for attr in layout.attrs):
            ctx.error(
                call,
                f"input {index} has dynamic shard ownership; use an explicit Reshard before {op_name}",
            )


def require_compatible_meshes(ctx, call, types, op_name: str) -> None:
    """Require all placed inputs to reference one mesh."""
    placed = [
        (index, layout)
        for index, type_ in enumerate(types)
        if (layout := shard_layout_of(type_.layout)) is not None
    ]
    if not placed:
        return
    mesh = placed[0][1].mesh
    for index, layout in placed[1:]:
        if layout.mesh != mesh:
            ctx.error(
                call,
                f"input {index} references a different mesh; use an explicit Reshard before {op_name}",
            )


def require_uniform_partial_slices(ctx, call, types, output: ShardLayout, op_name: str) -> None:
    """Require every input to carry each Partial state derived for the output."""
    for mesh_axis, output_attr in enumerate(output.attrs):
        if not isinstance(output_attr, Partial):
            continue
        for index, type_ in enumerate(types):
            layout = shard_layout_of(type_.layout)
            input_attr = (
                layout.attrs[mesh_axis]
                if layout is not None
                and layout.mesh == output.mesh
                and mesh_axis < len(layout.attrs)
                else None
            )
            if input_attr != output_attr:
                ctx.error(
                    call,
                    f"input {index} does not carry {output_attr!r} on mesh axis "
                    f"{mesh_axis}; use an explicit Reshard before {op_name}",
                )


__all__ = [
    "reject_partials",
    "check_multilinear_partials",
    "require_matching_partial_state",
    "reject_dynamic_shards",
    "require_compatible_meshes",
    "require_uniform_partial_slices",
]
