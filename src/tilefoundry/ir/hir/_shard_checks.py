"""Shared shard-soundness typeinfer checks, used across HIR op categories.

Three recurring Partial-commutation rules, factored out of the per-op
typeinfer bodies that enforce them (each op keeps only its own commute-set
constants):

- ``reject_partials``: the elementwise rule — a Partial reduction on a single
  operand commutes only when it is in ``commutes_with``.
- ``check_multilinear_partials``: the multilinear/bilinear rule — at most one
  operand may carry a Partial per mesh axis (in ``allowed_reduction``), and
  every other operand must be Broadcast/replicated on that axis.
- ``require_matching_partial_state``: the dst/update rule — an in-place-style
  op's two same-shape operands must carry identical per-mesh-axis states.
"""
from __future__ import annotations

from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout
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
    if not isinstance(layout, ShardLayout) or axis >= len(layout.attrs):
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
    """Multilinear-combine Partial-commutation check over several operands.

    On each mesh axis, at most one of *named_operands* (``(name, TensorType)``
    pairs) may carry a Partial state: its reduction must be in
    *allowed_reduction* (a single reduction name or a set of names), and every
    other operand must be Broadcast/replicated on that axis. Several operands
    may carry a Partial together when they all share one of
    *commutes_jointly*'s reductions (e.g. elementwise ADD of two
    ``Partial("sum")`` operands). *anchor*, when given, is the only operand
    whose Partial the output can preserve — a Partial on any other operand is
    rejected outright, regardless of its reduction.
    """
    allowed = (
        {allowed_reduction} if isinstance(allowed_reduction, str) else set(allowed_reduction)
    )
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
    """Require *dst* and *update* to carry the identical per-mesh-axis Partial
    state (an in-place-style write can only merge two operands whose shard
    states already agree)."""
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
    if dst_partials:
        if not (
            isinstance(dst.layout, ShardLayout)
            and isinstance(update.layout, ShardLayout)
            and update.layout.mesh == dst.layout.mesh
            and update.layout.attrs == dst.layout.attrs
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


__all__ = [
    "reject_partials",
    "check_multilinear_partials",
    "require_matching_partial_state",
]
