"""The public Schedule boundary.

One call names a program and a level of its parallel hierarchy; one registered
algorithm answers with a plan it owns entirely. The names re-exported here are
that boundary and nothing else: how an algorithm reaches its answer, and what it
looks at on the way, are its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .api import ScheduleResult, schedule
from .errors import ScheduleError
from .plan import PlanVerificationError, SchedulePlan


@dataclass(frozen=True)
class ScheduleOptions:
    """Solver runtime and debug controls, independent of which algorithm runs.

    `stop_at_first_solution` asks for a plan rather than the best plan. The search
    minimises a makespan, so on a model it cannot prove optimal for it keeps
    improving until the time limit -- which makes `timeout_seconds` the runtime of
    every solve rather than a limit that rarely fires. A caller that needs a plan to
    exist and to verify, and not to be optimal, says so here and gets the first
    feasible one. The time limit still applies: a model with no solution found yet
    is still bounded, so this cannot turn a slow search into an unbounded one.
    """

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    stop_at_first_solution: bool = False
    debug_dump_dir: Path | None = None


__all__ = [
    "PlanVerificationError",
    "ScheduleError",
    "ScheduleOptions",
    "SchedulePlan",
    "ScheduleResult",
    "schedule",
]
