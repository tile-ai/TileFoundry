"""The immutable scheduler descriptor selected by a Target."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# An algorithm solves for the selected Module and Function at one topology.
ScheduleCallable = Callable[
    ["Module", "Function", "Target", "Topology", object | None],
    "SchedulePlan",
]


@dataclass(frozen=True)
class Scheduler:
    """One scheduler service: the level it solves for, and the solve."""

    topology: str
    solve: ScheduleCallable

    def __post_init__(self) -> None:
        if not self.topology:
            raise ValueError("a schedule algorithm needs a non-empty topology name")


__all__ = [
    "ScheduleCallable",
    "Scheduler",
]
