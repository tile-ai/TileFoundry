"""Schedule obtains one solver from the exact Target value."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule.api import ScheduleResult, schedule
from tilefoundry.schedule.errors import ScheduleError
from tilefoundry.schedule.plan import PlanVerificationError, SchedulePlan
from tilefoundry.target import CudaTarget, Target, TopologyLimitFacts, register_target
from tilefoundry.target.services import Scheduler


@dataclass(frozen=True)
class _Plan(SchedulePlan):
    width: int
    valid: bool = True

    def verify(self, module, function, topology) -> None:
        if not self.valid:
            raise PlanVerificationError("invalid plan")

    def to_json(self) -> str:
        return json.dumps({"width": self.width})

    def render(self) -> str:
        return str(self.width)


_CALLS: list[str] = []


def _solve(module, function, target, topology, options):
    _CALLS.append(topology.name)
    return _Plan(topology.size)


@dataclass(frozen=True)
class _TopologyTarget(Target):
    topology_levels: ClassVar[tuple[str, ...]] = ("tile",)

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is TopologyLimitFacts and query == "tile":
            return TopologyLimitFacts("tile", 4)
        return super().get_facts(facts_type, query)


@dataclass(frozen=True)
class _SchedulingTarget(_TopologyTarget):
    name: ClassVar[str] = "test.scheduler"

    def get_scheduler(self, topology: str) -> Scheduler:
        if topology == "tile":
            return Scheduler("tile", _solve)
        return super().get_scheduler(topology)


@dataclass(frozen=True)
class _UnsupportedTarget(_TopologyTarget):
    name: ClassVar[str] = "test.unsupported-scheduler"


@func
def scale(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    return relu(x)  # noqa: F405


@module(entry="scale", target=_SchedulingTarget())
class Widget:
    topologies = (Topology("tile", 4),)
    scale = scale


@module(entry="scale", target=_UnsupportedTarget())
class Unsupported:
    topologies = (Topology("tile", 4),)
    scale = scale


@dataclass(frozen=True)
class _BrokenSchedulerTarget(_TopologyTarget):
    name: ClassVar[str] = "test.broken-scheduler"

    def get_scheduler(self, topology: str) -> Scheduler:
        raise ValueError("provider scheduler failure")


@module(entry="scale", target=_BrokenSchedulerTarget())
class BrokenScheduler:
    topologies = (Topology("tile", 4),)
    scale = scale


@module(entry="scale", target=CudaTarget("nvidia.h200_sxm"))
class OverLimitCuda:
    topologies = (Topology("cta", 1), Topology("thread", 1025))
    scale = scale


def test_schedule_uses_the_target_selected_solver() -> None:
    _CALLS.clear()
    result = schedule(Widget, scale, topology="tile")

    assert isinstance(result, ScheduleResult)
    assert result.plan == _Plan(4)
    assert _CALLS == ["tile"]


def test_an_unsupported_scheduler_fails_before_a_solver_runs() -> None:
    _CALLS.clear()
    with pytest.raises(ScheduleError, match="Target .*no scheduler for 'tile'"):
        schedule(Unsupported, scale, topology="tile")
    assert _CALLS == []


def test_schedule_preserves_plan_verification() -> None:
    @register_target
    @dataclass(frozen=True)
    class _InvalidTarget(_SchedulingTarget):
        name: ClassVar[str] = "test.invalid-scheduler"

        def get_scheduler(self, topology: str) -> Scheduler:
            return Scheduler(topology, lambda *_args: _Plan(4, valid=False))

    @module(entry="scale", target=_InvalidTarget())
    class Invalid:
        topologies = (Topology("tile", 4),)
        scale = scale

    with pytest.raises(PlanVerificationError, match="invalid plan"):
        schedule(Invalid, scale, topology="tile")


def test_schedule_keeps_provider_failures_and_rejects_over_limit_topology() -> None:
    with pytest.raises(ValueError, match="provider scheduler failure"):
        schedule(BrokenScheduler, scale, topology="tile")
    with pytest.raises(ValueError, match="1 <= extent <= 1024"):
        schedule(OverLimitCuda, scale, topology="thread")
