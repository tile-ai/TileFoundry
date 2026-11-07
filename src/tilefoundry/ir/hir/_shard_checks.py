"""Enforce shared HIR shard-soundness rules during type inference.

Checks cover elementwise commutation, multilinear combinations, and matching
Partial state for in-place-style operations.

See [shard §8](docs/spec/shard.md#8-layout-propagation).
"""

from __future__ import annotations

from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    ShardAttr,
    ShardLayout,
    attrs_by_axis,
    partial_reductions_by_axis,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageOwners


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


def check_multilinear_partials(ctx, call, named_operands, *, kind=BinaryKind.MUL, anchor=None):
    """Check each Partial axis using the op's combination rule and storage owners."""
    layouts = [(name, ty, attrs_by_axis(ty.layout)) for name, ty in named_operands]
    target = getattr(ctx, "target", None)
    owners = target.storage_owners if target else StorageOwners((), {})
    for name, ty, layout in layouts:
        if layout is None:
            continue
        for axis, attr in layout.items():
            if not isinstance(attr, Partial):
                continue
            if anchor is not None and name != anchor:
                ctx.error(call, f"{name} carries {attr!r} on mesh axis {axis[1]} ({axis[0]}); "
                          f"reshard({name}, @ {{B()}}) before this consumer")
            state = None
            level = axis[0]
            for other_name, other_ty, other in layouts:
                other_attr = (
                    other.get(axis, Broadcast()) if other is not None else None
                )
                held = ShardAttr.of(other_attr, owners.private(other_ty.storage, level))
                try:
                    state = held if state is None else ShardAttr.combine(kind, state, held)
                except ValueError as error:
                    ctx.error(call, f"{name} carries Partial({attr.reduction}) on mesh axis {axis[1]} ({axis[0]}); "
                              f"operand {other_name}: operands do not commute: {error}")


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
    "require_uniform_partial_slices",
]
