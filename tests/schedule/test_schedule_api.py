"""``schedule(module, function, topology=...)`` -- what the public call decides
and what it refuses to guess.

The subject is the boundary, not a solver. A scheduler is chosen by the exact
pair of hardware and hierarchy level the program declares, the plan it returns
is entirely its own, and every way of asking for something that does not exist
is answered before an algorithm is allowed to run.

The target and the level names here are local to this file on purpose: the
contract under test is that dispatch is exact, which is a statement about the
lookup rather than about any particular machine's vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- relu resolved dynamically
from tilefoundry.ir.types.shard import Topology
from tilefoundry.registry import DuplicateAlgorithmError
from tilefoundry.schedule import ScheduleOptions
from tilefoundry.schedule.api import ScheduleResult, schedule
from tilefoundry.schedule.errors import ScheduleError
from tilefoundry.schedule.plan import PlanVerificationError, SchedulePlan
from tilefoundry.schedule.registry import SCHEDULES, register_schedule
from tilefoundry.target.base import Target


@dataclass(frozen=True)
class _WidgetTarget(Target):
    """A target whose only purpose is to own a registration in this file."""

    name: str = field(default="widget", init=False)


@dataclass(frozen=True)
class _WidgetPlusTarget(_WidgetTarget):
    """A subtype of a registered target, to show it inherits no scheduler."""

    name: str = field(default="widget-plus", init=False)


@dataclass(frozen=True)
class _GadgetTarget(Target):
    """A second target, so the support matrix has a hole to observe."""

    name: str = field(default="gadget", init=False)


@dataclass(frozen=True)
class _WidgetPlan(SchedulePlan):
    """The whole plan of the scheduler below: which level, how wide."""

    level: str
    width: int
    consistent: bool = True

    def verify(self, module, function, topology) -> None:
        if not self.consistent:
            raise PlanVerificationError(
                f"widget plan claims width {self.width} at {self.level!r}"
            )

    def to_json(self) -> str:
        return json.dumps({"level": self.level, "width": self.width}, sort_keys=True)

    def render(self) -> str:
        return f"{self.level} x{self.width}"


_CALLS: list[tuple[str, int]] = []
_OPTIONS: list[ScheduleOptions] = []


@func
def scale(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    h = relu(x)  # noqa: F405
    return h


@func
def offset(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    h = relu(x)  # noqa: F405
    return h


@module(entry="scale", target=_WidgetTarget())
class Widget:
    topologies = (Topology("tile", 4), Topology("slot", 8))
    scale = scale


@module(entry="scale", target=_GadgetTarget())
class Gadget:
    topologies = (Topology("tile", 4),)
    scale = scale


@module(entry="scale", target=_WidgetPlusTarget())
class WidgetPlus:
    topologies = (Topology("tile", 4),)
    scale = scale


@module(entry="scale", target=_WidgetTarget())
class Launched:
    topologies = (Topology("cta", None),)
    scale = scale


@register_schedule(_WidgetTarget, "tile")
def _solve_widget_tile(module, function, target, topology, options):
    _CALLS.append((topology.name, topology.size))
    assert isinstance(options, ScheduleOptions)
    _OPTIONS.append(options)
    return _WidgetPlan(level=topology.name, width=topology.size)


@register_schedule(_WidgetTarget, "slot")
def _solve_widget_slot(module, function, target, topology, options):
    _CALLS.append((topology.name, topology.size))
    return _WidgetPlan(level=topology.name, width=topology.size, consistent=False)


@register_schedule(_WidgetTarget, "cta")
def _solve_widget_cta(module, function, target, topology, options):
    _CALLS.append((topology.name, topology.size))
    return _WidgetPlan(level=topology.name, width=1)


@pytest.fixture(autouse=True)
def _clear_calls():
    _CALLS.clear()
    _OPTIONS.clear()


def test_schedule_returns_its_inputs_and_the_resolved_level():
    """One call dispatches once and reports what it decided against.

    The Module and Function come back as the same objects: scheduling decides
    about a program rather than producing a new one. The topology comes back
    resolved, so the caller holds the extent the algorithm actually saw instead
    of the name it asked by.
    """
    result = schedule(Widget, scale, topology="tile")

    assert isinstance(result, ScheduleResult)
    assert result.module is Widget
    assert result.function is scale
    assert result.topology is Widget.effective_topologies()[0]
    assert result.topology == Topology("tile", 4)
    assert _CALLS == [("tile", 4)]

    assert result.plan.render() == "tile x4"
    assert json.loads(result.plan.to_json()) == {"level": "tile", "width": 4}


def test_schedule_normalizes_options_before_the_algorithm_runs():
    """Every scheduler receives one fresh common options value per default call."""
    schedule(Widget, scale, topology="tile")
    schedule(Widget, scale, topology="tile")

    assert len(_OPTIONS) == 2
    assert _OPTIONS[0] is not _OPTIONS[1]

    with pytest.raises(ScheduleError, match="options must be ScheduleOptions"):
        schedule(Widget, scale, topology="tile", options=object())
    assert _CALLS == [("tile", 4), ("tile", 4)]


def test_dispatch_is_exact_in_both_target_and_level():
    """Neither half of the key is guessed.

    A level the target has no scheduler for is not served by the target's other
    levels, a target with no scheduler at all is not served by a target that has
    one, and a subclass does not inherit its base's registration -- two machines
    that share a base can need different schedulers.
    """
    with pytest.raises(ScheduleError, match="no 'tile' registered for _GadgetTarget"):
        schedule(Gadget, scale, topology="tile")

    with pytest.raises(
        ScheduleError, match="no 'tile' registered for _WidgetPlusTarget"
    ):
        schedule(WidgetPlus, scale, topology="tile")

    assert _CALLS == []


def test_registering_the_same_pair_twice_is_refused():
    """The support matrix is what the registrations say, so it stays single-valued."""
    with pytest.raises(DuplicateAlgorithmError, match="'tile' is already registered"):
        register_schedule(_WidgetTarget, "tile")(_solve_widget_tile)


def test_the_registry_reports_the_levels_it_serves():
    """A missing pair is diagnosed with what is available, not just what failed."""
    assert SCHEDULES.selectors_for(_WidgetTarget) == ("cta", "slot", "tile")
    assert SCHEDULES.selectors_for(_GadgetTarget) == ()


@pytest.mark.parametrize(
    ("target_module", "level", "message"),
    [
        (Widget, "warp", "no topology named 'warp'"),
        (Launched, "cta", "not known until launch"),
        (Widget, "", "non-empty level name"),
    ],
)
def test_an_unservable_request_fails_before_any_algorithm_runs(
    target_module, level, message
):
    """Resolution finishes first, so a refusal never has a partial solve behind it.

    A level the program does not declare and a level sized at launch are both
    answered from the declaration. The launch-provided case is the interesting
    one: the pair is registered and the level exists, and it still fails,
    because placing work across a level means counting it.

    A name that maps to two levels is not among these, because a Module cannot
    declare one twice -- ambiguity is refused where the hierarchy is written
    rather than where it is read.
    """
    with pytest.raises(ScheduleError, match=message):
        schedule(target_module, scale, topology=level)

    assert _CALLS == []


def test_a_hierarchy_cannot_name_one_level_twice():
    """The declaration is single-valued, so no level name is ever ambiguous."""
    with pytest.raises(ValueError, match="duplicate topology name"):

        @module(entry="scale", target=_WidgetTarget())
        class Doubled:
            topologies = (Topology("tile", 4), Topology("tile", 8))
            scale = scale


def test_a_function_outside_the_module_is_not_scheduled():
    """The Module is the execution domain, so the Function has to be part of it."""
    with pytest.raises(ScheduleError, match="not a function of module 'Widget'"):
        schedule(Widget, offset, topology="tile")

    assert _CALLS == []


def test_a_function_alone_is_not_an_execution_domain():
    """A Function declares neither hardware nor a hierarchy, so it cannot stand in."""
    with pytest.raises(TypeError, match="expected a Module"):
        schedule(scale, scale, topology="tile")

    with pytest.raises(TypeError, match="expected an hir.Function"):
        schedule(Widget, Widget, topology="tile")


def test_a_plan_that_does_not_hold_together_does_not_reach_the_caller():
    """Verification runs on the way out, so a result always carries a checked plan.

    The algorithm ran and returned, which is what separates this from the
    resolution failures: the request was servable and the answer was not.
    """
    with pytest.raises(PlanVerificationError, match="claims width 8"):
        schedule(Widget, scale, topology="slot")

    assert _CALLS == [("slot", 8)]


def test_the_plan_base_promises_only_the_three_operations():
    """The base names what a plan must do and nothing about what it contains.

    There is no shared schema and no way back from text: a plan is produced by
    the algorithm that solved for it, so reading one in would mean trusting a
    document to describe decisions nobody made.
    """
    assert not hasattr(SchedulePlan, "to_data")
    assert not hasattr(SchedulePlan, "from_json")
    assert not hasattr(SchedulePlan, "version")

    base = SchedulePlan()
    for call in (
        lambda: base.verify(Widget, scale, Topology("tile", 4)),
        base.to_json,
        base.render,
    ):
        with pytest.raises(NotImplementedError):
            call()
