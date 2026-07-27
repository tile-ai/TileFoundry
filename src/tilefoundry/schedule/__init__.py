"""The public Schedule boundary.

One call names a program and a level of its parallel hierarchy; one registered
algorithm answers with a plan it owns entirely. The names re-exported here are
that boundary and nothing else: how an algorithm reaches its answer, and what it
looks at on the way, are its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True)
class ScheduleReport:
    """What a solve proved about its objective.

    This is the part of an answer that does not depend on what was decided: how
    good the result is, and how sure the solver is of it. An algorithm whose
    plan states an objective carries one of these rather than restating the same
    nine fields in its own vocabulary.
    """

    root: str
    target: str
    topology: str
    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_name: Literal["makespan"]
    unit: Literal["ns"]
    selected: int
    best_bound: int
    gap: float

    def to_json(self) -> str:
        """Render the complete summary as sorted-key JSON."""
        return json.dumps(asdict(self), sort_keys=True)

    def to_markdown(self) -> str:
        """Render the complete summary as a stable Markdown table."""
        rows = (
            ("root", self.root),
            ("target", self.target),
            ("topology", self.topology),
            ("status", self.status),
            ("objective_name", self.objective_name),
            ("unit", self.unit),
            ("selected", self.selected),
            ("best_bound", self.best_bound),
            ("gap", self.gap),
        )
        lines = ["| field | value |", "| --- | --- |"]
        lines.extend(f"| {field} | {value} |" for field, value in rows)
        return "\n".join(lines)


__all__ = [
    "PlanVerificationError",
    "ScheduleError",
    "ScheduleOptions",
    "SchedulePlan",
    "ScheduleReport",
    "ScheduleResult",
    "schedule",
]
