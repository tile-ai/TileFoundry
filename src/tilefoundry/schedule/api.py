"""Stage-agnostic schedule invocation contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from tilefoundry.schedule.report import ScheduleReport

if TYPE_CHECKING:
    from tilefoundry.ir.core.module import Module
    from tilefoundry.ir.hir.function import Function


@dataclass(frozen=True)
class ScheduleOptions:
    """Solver runtime and debug controls shared by every stage."""

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    debug_dump_dir: Path | None = None


class Schedule(Protocol):
    """One stage's complete solve service."""

    stage: str

    def solve(
        self,
        module: Module,
        root: Function,
        options: ScheduleOptions,
    ) -> ScheduleResult: ...


@dataclass(frozen=True)
class ScheduleResult:
    """A materialized module and its public summary report."""

    module: Module
    report: ScheduleReport


class ScheduleError(Exception):
    """An actionable scheduling failure exposed to callers."""


def solve(
    module: Module,
    *,
    root: Function,
    stage: str,
    options: ScheduleOptions | None = None,
) -> ScheduleResult:
    """Dispatch an exact stage request through the root Function's Target."""
    if not any(root is function for function in module.functions):
        raise ScheduleError(
            f"schedule.solve: root {getattr(root, 'name', root)!r} is not one "
            "of module.functions"
        )
    if not isinstance(stage, str) or not stage:
        raise ScheduleError(
            f"schedule.solve: stage must be a non-empty str, got {stage!r}"
        )
    if options is not None and not isinstance(options, ScheduleOptions):
        raise ScheduleError(
            "schedule.solve: options must be ScheduleOptions or None, "
            f"got {type(options).__name__}"
        )

    schedule = root.target.service(Schedule, stage)
    return schedule.solve(module, root, options if options is not None else ScheduleOptions())


__all__ = [
    "Schedule",
    "ScheduleError",
    "ScheduleOptions",
    "ScheduleResult",
    "solve",
]
