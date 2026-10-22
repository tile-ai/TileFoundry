"""Construction of isl iteration domains from authored loop bounds."""

from __future__ import annotations

import isl

from tilefoundry.ir.core import value_label
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.visitor_registry.access_relation import AccessRelation

from .errors import AnalysisError


def induction_name(loop: LoopRegion) -> str:
    """Name a loop by the induction variable authored for it."""
    return getattr(loop.induction_var, "name", None) or "<unnamed>"


def refuse_runtime_bound(loop: LoopRegion, which: str) -> None:
    """Reject a loop bound that has no literal value or stated range."""
    value = getattr(loop, which)
    if which == "extent":
        raise AnalysisError(
            f"loop {induction_name(loop)!r} has a trip count the program computes "
            f"at run time from {value_label(value) or 'a value'!r}, so no "
            f"per-occurrence total can be scaled by it; bind the extent to a "
            f"literal, or state it as an open dimension"
        )
    raise AnalysisError(
        f"loop {induction_name(loop)!r} takes its {which} from "
        f"{value_label(value) or 'a value'!r}, which the program computes at run time; "
        f"analysis needs a literal {which} or a stated value range"
    )


def iteration_domain(owner, parent, *, first=False, loops=None):
    """Represent authored iteration coordinates as identity-named parameters."""
    if loops is None:
        loops = tuple(s.owner for s in parent.loop_scopes) if parent is not None else ()
        if isinstance(owner, LoopRegion):
            loops = (*loops, owner)
    else:
        loops = tuple(s.owner for s in loops)
    for loop in loops:
        if static_dim_value(loop.step) is None:
            refuse_runtime_bound(loop, "step")
    try:
        return loop_domain(
            tuple((loop.induction_var, loop.start, loop.extent, loop.step) for loop in loops),
            {}, first=first,
        )
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        for loop in loops:
            try:
                AccessRelation.affine_of(loop.start, parent)
                AccessRelation.affine_of(loop.extent, parent)
            except (TypeError, ValueError, NotImplementedError, isl.Error):
                refuse_runtime_bound(loop, "extent")
        raise


def loop_domain(loops, values, *, first=False):
    """Constrain named induction values by the shared half-open loop bounds."""
    result = AccessRelation(isl.map("{ [] -> [] }"))
    for induction_var, begin, end, stride in loops:
        start = AccessRelation.affine_of(begin, values=values)
        stop = AccessRelation.affine_of(end, values=values)
        step = static_dim_value(stride)
        if step is None:
            raise ValueError("loop step must be static")
        name = AccessRelation.param_name(induction_var)
        values[name] = induction_var
        induction = isl.pw_aff(f"[{name}] -> {{ [{name}] }}")
        if first:
            domain = start.eq_set(induction)
        else:
            domain = start.le_set(induction).intersect(induction.lt_set(stop))
            if step > 1:
                domain = domain.intersect(induction.sub(start).mod(isl.val(step)).zero_set())
        result = result.constrain(domain.params(), values)
    return result


__all__ = ["induction_name", "iteration_domain", "loop_domain", "refuse_runtime_bound"]
