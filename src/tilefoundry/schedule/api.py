"""The public Schedule operation.

One call names one Module, one Function, and one level of the parallel
hierarchy, and gets back the plan one algorithm produced for exactly that
combination. Everything the caller has to decide is in the call; everything the
algorithm decided is in the plan.

The Module is the execution domain: it declares the hardware and the ordered
topology hierarchy, so a level is resolved against the program that declares it
rather than against a name the caller invents. Resolution and dispatch both
finish before any algorithm runs, so a request that cannot be served fails
without a partial solve to explain.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import Topology
from tilefoundry.registry import UnknownAlgorithmError

from .errors import ScheduleError
from .plan import SchedulePlan
from .registry import SCHEDULES, ScheduleAlgorithm


@dataclass(frozen=True)
class ScheduleResult:
    """What one public Schedule call decided, and what it decided it against.

    The Module and Function are the same objects that were passed in. Scheduling
    is a decision about a program, not a rewrite of one, so there is nothing
    here that the caller would have to diff against its own input to recognise.
    """

    module: Module
    function: Function
    topology: Topology
    plan: SchedulePlan


def _topology(module: Module, name: str) -> Topology:
    """The one level of *module*'s hierarchy called *name*.

    A level whose extent arrives at launch is rejected: an algorithm places work
    across a level by counting it, and there is nothing to count yet. What comes
    back is the level the program declared, not a normalized copy of it, so the
    caller sees its own hierarchy rather than one this layer rewrote.
    """
    if not isinstance(name, str) or not name:
        raise ScheduleError(
            f"schedule: topology must be a non-empty level name, got {name!r}"
        )
    try:
        level = module.resolve_topology(name)
    except ValueError as error:
        raise ScheduleError(f"schedule: {error}") from None
    extent = static_dim_value(level.size)
    if extent is None:
        raise ScheduleError(
            f"schedule: topology {name!r} has extent {level.size!r}, which is "
            "not known until launch; scheduling a level requires its static "
            "extent"
        )
    if extent < 1:
        raise ScheduleError(
            f"schedule: topology {name!r} extent {extent} must be positive"
        )
    return level


def _algorithm(target: object, topology: str) -> ScheduleAlgorithm:
    """The algorithm bound to *topology* under *target*'s exact type."""
    try:
        return SCHEDULES.resolve(target, topology)
    except UnknownAlgorithmError as error:
        raise ScheduleError(f"schedule: {error}") from None


def _options(options: object | None) -> object:
    """The common options every registered algorithm receives."""
    # Importing through the package is safe at call time and keeps the public
    # value type in its existing package boundary without an import cycle.
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
) -> ScheduleResult:
    """Solve *function* at the *topology* level of *module*'s hierarchy."""
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
    if function not in module.functions:
        raise ScheduleError(
            f"schedule: {function.name!r} is not a function of module "
            f"{module.name!r}"
        )

    target = module.resolve_target()
    level = _topology(module, topology)
    algorithm = _algorithm(target, topology)
    resolved_options = _options(options)

    plan = algorithm.solve(module, function, target, level, resolved_options)
    if not isinstance(plan, SchedulePlan):
        raise ScheduleError(
            f"schedule: the {topology!r} algorithm for "
            f"{type(target).__name__} returned a {type(plan).__name__}, not a "
            "SchedulePlan"
        )
    plan.verify(module, function, level)
    return ScheduleResult(
        module=module, function=function, topology=level, plan=plan
    )


__all__ = ["ScheduleResult", "schedule"]
