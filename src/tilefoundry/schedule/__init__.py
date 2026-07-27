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
    """Solver runtime and debug controls, independent of which algorithm runs."""

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    debug_dump_dir: Path | None = None


__all__ = [
    "PlanVerificationError",
    "ScheduleError",
    "ScheduleOptions",
    "SchedulePlan",
    "ScheduleResult",
    "schedule",
]
