"""Private CUDA CTA stage services."""

from __future__ import annotations

from pathlib import Path

from tilefoundry.analysis import AtomFact
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.schedule import ScheduleOptions, ScheduleResult

from .atoms import candidate_atoms
from .materialize import materialize_planning_solution
from .planner import build_planning_problem
from .report import project_schedule_report
from .solver import solve_planning_problem


class _CudaCtaAnalysis:
    stage = "cta"

    def __init__(self, target: "CudaTarget") -> None:
        self._target = target

    @property
    def tile_capacity_bytes(self) -> int:
        """A CTA-level tile's resident working set lives in shared memory."""
        return self._target.device.shared_memory_per_cta_bytes

    def candidate_atoms(self, op: Call) -> list[AtomFact]:
        return candidate_atoms(op, self._target)


class _CudaCtaSchedule:
    stage = "cta"

    def __init__(self, target: "CudaTarget") -> None:
        self._target = target

    def solve(
        self,
        module: Module,
        root: Function,
        options: ScheduleOptions | None = None,
    ) -> ScheduleResult:
        if not isinstance(module, Module):
            raise TypeError(f"CTA Schedule expects a HIR Module, got {type(module).__name__}")
        if not isinstance(root, Function):
            raise TypeError(f"CTA Schedule expects a HIR Function root, got {type(root).__name__}")
        if root is not module.entry_function():
            raise ValueError("CTA Schedule requires root to be module.entry_function()")
        if root.target is not self._target:
            raise ValueError("CTA Schedule requires the root Target to own the requested service")
        if options is None:
            options = ScheduleOptions()
        if not isinstance(options, ScheduleOptions):
            raise TypeError(
                f"CTA Schedule options must be ScheduleOptions, got {type(options).__name__}"
            )
        problem = build_planning_problem(module, root)
        solution = solve_planning_problem(problem, options)
        materialized = materialize_planning_solution(problem, solution)
        verify_module(materialized.functions)
        if options.debug_dump_dir is not None:
            _write_materialized_dump(materialized, options.debug_dump_dir)
        report = project_schedule_report(problem, solution, stage=self.stage)
        return ScheduleResult(module=materialized, report=report)


def _write_materialized_dump(module: Module, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    source = as_script(module)
    (directory / "materialized_hir.py").write_text(source)


__all__ = ["_CudaCtaAnalysis", "_CudaCtaSchedule"]
