"""Construction of isl iteration domains from authored loop bounds."""

from __future__ import annotations

import isl

from tilefoundry.ir.core import value_label
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.isl_interop import IslParamValues, dim_to_isl_pw_aff
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.utils.isl_utils import has_unbounded_param

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


def _bound_on(
    loop: LoopRegion, which: str, values: IslParamValues, coords: dict[int, str]
) -> isl.pw_aff:
    """One start or extent as a function of the enclosing induction variables."""
    value = getattr(loop, which)
    number = static_dim_value(value)
    try:
        bound = dim_to_isl_pw_aff(number if number is not None else value, values, coords=coords)
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        refuse_runtime_bound(loop, which)
    if has_unbounded_param(bound):
        refuse_runtime_bound(loop, which)
    return bound


def iteration_domain(
    owner: Function | LoopRegion, parent: "IterationScope | None"
) -> tuple[isl.set, IslParamValues]:
    """Build the accumulated authored iteration domain for one scope owner."""
    if isinstance(owner, Function):
        return isl.set("{ [] }"), {}
    loops = (*(() if parent is None else parent.enclosing_loops()), owner)
    names = [f"p{index}" for index in range(len(loops))]
    coords = {id(loop.induction_var): name for loop, name in zip(loops, names)}
    cursor = parent
    while cursor is not None:
        for param, _ in cursor.captures:
            root = parent.capture_root(param)
            if id(root) in coords:
                coords[id(param)] = coords[id(root)]
        cursor = cursor.parent
    values: IslParamValues = {}
    domain = isl.set(f"{{ [{', '.join(names)}] }}")
    for index, loop in enumerate(loops):
        start = _bound_on(loop, "start", values, coords)
        stop = _bound_on(loop, "extent", values, coords)
        step = static_dim_value(loop.step)
        if step is None:
            raise AnalysisError(
                f"loop {induction_name(loop)!r} takes its step from "
                f"{value_label(loop.step) or 'a value'!r}; analysis needs a literal "
                "step, because a parametric stride has no isl representation"
            )
        induction = isl.pw_aff(f"{{ [{', '.join(names)}] -> [p{index}] }}")
        domain = domain.intersect(start.le_set(induction)).intersect(induction.lt_set(stop))
        if step != 1:
            domain = domain.intersect(induction.sub(start).mod(isl.val(step)).zero_set())
    return domain, values


__all__ = [
    "induction_name",
    "iteration_domain",
    "refuse_runtime_bound",
]
