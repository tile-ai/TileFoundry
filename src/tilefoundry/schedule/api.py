"""Expose the public Schedule operation.

A call names a module, one of its functions, and a declared topology level. The
module owns hardware and topology resolution. Resolution and scheduler dispatch
finish before solving, and the returned plan records every algorithm decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tilefoundry.analysis.check import _resolve_program_geometry, check_program
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import SpecializationError
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import Target, UnsupportedCapabilityError
from tilefoundry.target.services import Scheduler

from .errors import ScheduleError
from .plan import SchedulePlan


@dataclass(frozen=True)
class ScheduleResult:
    """What one public Schedule call decided, and what it decided it against.

    Scheduling is a decision about a program, not a rewrite of one, so the
    Module is the one that was passed in.

    The Function is too, unless an extent was chosen for it -- then it is the
    concrete function derived from that input, because that is the program the
    plan was solved for and the one the plan can be verified against. A caller
    handed back its own symbolic input would have a plan it could not check.
    """

    module: Module
    function: Function
    topology: Topology
    plan: SchedulePlan


def _topology(module: Module, name: str) -> Topology:
    """Return one named level after the local scheduling preconditions.

    The launch-extent rule is defined by docs/spec/target.md § Topology levels.
    """
    if not isinstance(name, str) or not name:
        raise ScheduleError(
            f"schedule: topology must be a non-empty level name, got {name!r}"
        )
    try:
        level = module.resolve_topology(name)
    except ValueError as error:
        raise ScheduleError(f"schedule: {error}") from None
    return level


def _algorithm(target: Target, topology: str) -> Scheduler:
    """The scheduler selected by the resolved Target for *topology*."""
    try:
        return target.get_scheduler(topology)
    except UnsupportedCapabilityError as error:
        raise ScheduleError(f"schedule: {error}") from None


def _options(options: object | None) -> object:
    """The common options every registered algorithm receives."""
    from . import ScheduleOptions  # noqa: PLC0415

    if options is None:
        return ScheduleOptions()
    if not isinstance(options, ScheduleOptions):
        raise ScheduleError(
            "schedule: options must be ScheduleOptions, got "
            f"{type(options).__name__}"
        )
    return options


def schedule(
    module: Module,
    function: Function,
    *,
    topology: str,
    options: object | None = None,
    dims: "Mapping[str, int] | None" = None,
) -> ScheduleResult:
    """Solve *function* at the *topology* level of *module*'s hierarchy.

    *dims* selects concrete extents and the matching specialization before the
    solver counts work or capacity. The input must be a prototype or variant
    owned by *module*; substitution does not widen that ownership boundary.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"schedule: expected a Module, got {type(module).__name__}. A "
            "Function declares neither hardware nor a topology hierarchy; "
            "select the Module that owns it."
        )
    if not isinstance(function, Function):
        raise TypeError(
            f"schedule: expected an hir.Function, got {type(function).__name__}"
        )
    if not module.owns(function):
        raise ScheduleError(
            f"schedule: {function.name!r} is not a function of module "
            f"{module.name!r}"
        )
    result_module = module
    try:
        module, function = _resolve_program_geometry(module, function, dims)
    except SpecializationError as error:
        raise ScheduleError(f"schedule: {error}") from None

    target = module.resolve_target()
    level = _topology(module, topology)
    try:
        check_program(module, function, level=topology)
    except AnalysisError as error:
        raise ScheduleError(f"schedule: {error}") from None
    algorithm = _algorithm(target, topology)
    resolved_options = _options(options)

    plan = algorithm.solve(module, function, target, level, resolved_options)
    if not isinstance(plan, SchedulePlan):
        raise ScheduleError(
            f"schedule: the {topology!r} algorithm for "
            f"{type(target).__name__} returned a {type(plan).__name__}, not a "
            "SchedulePlan"
        )
    plan.verify(result_module, function, level)
    return ScheduleResult(
        module=result_module, function=function, topology=level, plan=plan
    )


__all__ = ["ScheduleResult", "schedule"]
