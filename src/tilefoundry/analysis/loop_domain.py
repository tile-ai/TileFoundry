"""Construction of isl iteration domains from authored loop bounds."""

from __future__ import annotations

import isl

from tilefoundry.ir.core import value_label
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types.dim_isl import dim_to_isl_expr
from tilefoundry.ir.types.shape_helpers import static_dim_value

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


def bound_to_isl_expr(
    loop: LoopRegion,
    which: str,
    params: dict[str, tuple[int, int] | None],
    param_map: dict[str, object],
    identities: dict[int, str],
) -> str:
    """Render one start or extent from bounded leaves into isl syntax."""
    value = getattr(loop, which)
    number = static_dim_value(value)
    if number is not None:
        return str(number)
    try:
        rendered = dim_to_isl_expr(
            value,
            params,
            param_map=param_map,
            identities=identities,
        )
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        refuse_runtime_bound(loop, which)
    if any(bound is None for bound in params.values()):
        refuse_runtime_bound(loop, which)
    return rendered


def iteration_domain(owner: Function | LoopRegion, parent: "IterationScope | None") -> isl.set:
    """Build the accumulated authored iteration domain for one scope owner."""
    if isinstance(owner, Function):
        return isl.set("{ [] }")
    loops = () if parent is None else parent.enclosing_loops()
    params: dict[str, tuple[int, int] | None] = {}
    param_map: dict[str, object] = {}
    identities: dict[int, str] = {}
    bounds: list[str] = []
    for index, loop in enumerate((*loops, owner)):
        start = bound_to_isl_expr(loop, "start", params, param_map, identities)
        stop = bound_to_isl_expr(loop, "extent", params, param_map, identities)
        step = static_dim_value(loop.step)
        if step is None:
            raise AnalysisError(
                f"loop {induction_name(loop)!r} takes its step from "
                f"{value_label(loop.step) or 'a value'!r}; analysis needs a literal "
                "step, because a parametric stride has no isl representation"
            )
        bounds.append(f"{start} <= p{index} < {stop}")
        if step != 1:
            bounds.append(f"(p{index} - {start}) mod {step} = 0")
    for name, bound in params.items():
        if bound is None:
            raise AnalysisError(f"loop domain parameter {name!r} has no stated value range")
        bounds.append(f"{bound[0]} <= {name} < {bound[1]}")
    names = ", ".join(f"p{index}" for index in range(len(loops) + 1))
    prefix = f"[{', '.join(params)}] -> " if params else ""
    return isl.set(f"{prefix}{{ [{names}] : {' and '.join(bounds)} }}")


__all__ = [
    "bound_to_isl_expr",
    "induction_name",
    "iteration_domain",
    "refuse_runtime_bound",
]
