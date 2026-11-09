"""Construction of isl iteration domains from authored loop bounds."""

from __future__ import annotations

import isl

from tilefoundry import services
from tilefoundry.interfaces import access
from tilefoundry.ir import core, expr

from .errors import Error


def induction_name(loop: core.LoopRegion) -> str:
    """Name a loop by the induction variable authored for it."""
    return getattr(loop.induction_var, "name", None) or "<unnamed>"


def span_bounds(loop: core.LoopRegion) -> tuple[int, int] | None:
    """Bounds on the least and greatest ``stop - start`` of *loop*; None without a stated range."""
    span = services.simplify_dim("sub", (loop.extent, loop.start))
    value = expr.static_dim_value(span)
    if value is not None:
        return value, value
    try:
        bounds = services.dim_range(span)
    except (ArithmeticError, TypeError, ValueError):
        return None
    return None if bounds is None else (bounds[0], bounds[1] - 1)


def refuse_runtime_bound(loop: core.LoopRegion, which: str) -> None:
    """Reject a loop bound that has no literal value or stated range."""
    value = getattr(loop, which)
    if which == "extent":
        raise Error(
            f"loop {induction_name(loop)!r} has a trip count the program computes "
            f"at run time from {expr.value_label(value) or 'a value'!r}, so no "
            f"per-occurrence total can be scaled by it; bind the extent to a "
            f"literal, or state it as an open dimension"
        )
    raise Error(
        f"loop {induction_name(loop)!r} takes its {which} from "
        f"{expr.value_label(value) or 'a value'!r}, which the program computes at run time; "
        f"analysis needs a literal {which} or a stated value range"
    )


def iteration_domain(owner, parent, *, first=False, loops=None):
    """Represent authored iteration coordinates as identity-named parameters."""
    if loops is None:
        loops = tuple(s.owner for s in parent.loop_scopes) if parent is not None else ()
        if isinstance(owner, core.LoopRegion):
            loops = (*loops, owner)
    else:
        loops = tuple(s.owner for s in loops)
    for loop in loops:
        if expr.static_dim_value(loop.step) is None:
            refuse_runtime_bound(loop, "step")
    try:
        return access.loop_domain(
            tuple((loop.induction_var, loop.start, loop.extent, loop.step) for loop in loops),
            {}, first=first,
        )
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        for loop in loops:
            try:
                access.affine_of(loop.start, parent)
                access.affine_of(loop.extent, parent)
            except (TypeError, ValueError, NotImplementedError, isl.Error):
                refuse_runtime_bound(loop, "extent")
        raise


__all__ = ["induction_name", "iteration_domain", "refuse_runtime_bound", "span_bounds"]
