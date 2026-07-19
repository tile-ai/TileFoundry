"""Direct public Schedule service contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class ScheduleOptions:
    """Solver runtime and debug controls shared by every stage."""

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    debug_dump_dir: Path | None = None


@dataclass(frozen=True)
class ScheduleReport:
    """Minimal objective summary shared by all schedule stages."""

    root: str
    target: str
    stage: str
    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_name: Literal["makespan"]
    unit: Literal["ns"]
    baseline: int
    selected: int
    solver_phase: Literal["makespan", "reshard_bytes", "resource_area"]
    proven_objectives: tuple[str, ...]
    best_bound: int | None
    gap: float | None

    def to_json(self) -> str:
        """Render the complete summary as sorted-key JSON."""
        return json.dumps(asdict(self), sort_keys=True)

    def to_markdown(self) -> str:
        """Render the complete summary as a stable Markdown table."""
        rows = (
            ("root", self.root),
            ("target", self.target),
            ("stage", self.stage),
            ("status", self.status),
            ("objective_name", self.objective_name),
            ("unit", self.unit),
            ("baseline", self.baseline),
            ("selected", self.selected),
            ("solver_phase", self.solver_phase),
            ("proven_objectives", ", ".join(self.proven_objectives)),
            ("best_bound", self.best_bound),
            ("gap", self.gap),
        )
        lines = ["| field | value |", "| --- | --- |"]
        lines.extend(f"| {field} | {value} |" for field, value in rows)
        return "\n".join(lines)


@dataclass(frozen=True)
class ScheduleResult:
    """A materialized module and its public summary report."""

    module: "Module"
    report: ScheduleReport


class Schedule(Protocol):
    """One stage's complete solve service."""

    stage: str

    def solve(
        self,
        module: "Module",
        root: "Function",
        options: ScheduleOptions,
    ) -> ScheduleResult: ...

__all__ = [
    "Schedule",
    "ScheduleOptions",
    "ScheduleResult",
    "ScheduleReport",
]
