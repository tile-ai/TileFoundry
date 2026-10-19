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
    result = AccessRelation(isl.map("{ [] -> [] }"))
    values = {}
    for loop in loops:
        try:
            start = AccessRelation.affine(loop.start, parent, values=values)
            stop = AccessRelation.affine(loop.extent, parent, values=values)
        except (TypeError, ValueError, NotImplementedError, isl.Error):
            refuse_runtime_bound(loop, "extent")
        step = static_dim_value(loop.step)
        if step is None:
            refuse_runtime_bound(loop, "step")
        name = AccessRelation.param_name(loop.induction_var)
        values[name] = loop.induction_var
        induction = isl.pw_aff(f"[{name}] -> {{ [{name}] }}")
        if first:
            domain = start.eq_set(induction)
        else:
            domain = start.le_set(induction).intersect(induction.lt_set(stop))
            if step > 1:
                domain = domain.intersect(induction.sub(start).mod(isl.val(step)).zero_set())
        result = result.constrain(domain.params(), values)
    return result


__all__ = ["induction_name", "iteration_domain", "refuse_runtime_bound"]
