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

from collections.abc import Mapping
from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import SpecializationError, specialize_concretely
from tilefoundry.ir.types.shape_helpers import static_dim_value
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
    extent = static_dim_value(level.size)
    if extent is None:
        raise ScheduleError(
            f"schedule: topology {name!r} has extent {level.size!r}, which is "
            "not known until launch; scheduling a level requires its static "
            "extent. The rule: tilefoundry spec target topology-levels"
        )
    if extent < 1:
        raise ScheduleError(
            f"schedule: topology {name!r} extent {extent} must be positive"
        )
    return level


def _algorithm(target: Target, topology: str) -> Scheduler:
    """The scheduler selected by the resolved Target for *topology*."""
    try:
        return target.get_scheduler(topology)
    except UnsupportedCapabilityError as error:
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
    dims: "Mapping[str, int] | None" = None,
) -> ScheduleResult:
    """Solve *function* at the *topology* level of *module*'s hierarchy.

    *dims* states an extent for each dimension the function declares as a
    range. A solver lays work across a level by counting it and holds a tile
    against a capacity in bytes, so a function authored for many context lengths
    is solved at one of them: the variant covering that length is resolved and
    its ranges substituted, and the plan is a plan for that size.

    The function passed in must still be one this Module owns -- a prototype or
    one of its variants. Substitution happens after that, so nothing loosens
    which programs a Module will answer for.
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
    if dims is not None:
        try:
            function = specialize_concretely(function, dims)
        except SpecializationError as error:
            raise ScheduleError(f"schedule: {error}") from None

    target = module.resolve_target()
    for declared_topology in module.effective_topologies():
        target.validate_program_topology(declared_topology)
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
