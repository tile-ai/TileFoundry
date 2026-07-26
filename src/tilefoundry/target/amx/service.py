"""Private AMX core stage services."""

from __future__ import annotations

from tilefoundry.analysis import AtomFact, extract
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule import ScheduleOptions, ScheduleReport, ScheduleResult
from tilefoundry.schedule.kernel_schedule import build_schedule_tree

# `select_atoms` measures its makespan in sub-ns duration units; the public
# report is in ns, and the scale that converts them has no public accessor yet.
from tilefoundry.schedule.select_atoms import _DURATION_SCALE, select_atoms

from .atoms import candidate_atoms


class _AmxCoreAnalysis:
    stage = "core"

    def __init__(self, target: "AmxTarget") -> None:
        self._target = target

    @property
    def tile_capacity_bytes(self) -> int:
        """A core-level tile's resident working set lives in that core's L1d.
        The AMX register files bound one atom instance, not a tile, and do it
        by filtering that atom out of ``candidate_atoms``."""
        return self._target.device.l1d_bytes_per_performance_core

    def candidate_atoms(self, op: Call) -> list[AtomFact]:
        return candidate_atoms(op, self._target)


class _AmxCoreSchedule:
    stage = "core"

    def __init__(self, target: "AmxTarget") -> None:
        self._target = target

    def solve(
        self,
        module: Module,
        root: Function,
        options: ScheduleOptions | None = None,
    ) -> ScheduleResult:
        """Decide resources over ``root``'s schedule tree and report the
        objective. Nothing is materialized at this stage, so the returned
        module is the one that was passed in.
        """
        if not isinstance(module, Module):
            raise TypeError(
                f"core Schedule expects a HIR Module, got {type(module).__name__}"
            )
        if not isinstance(root, Function):
            raise TypeError(
                f"core Schedule expects a HIR Function root, got {type(root).__name__}"
            )
        if root is not module.entry_function():
            raise ValueError("core Schedule requires root to be module.entry_function()")
        if root.target is not self._target:
            raise ValueError(
                "core Schedule requires the root Target to own the requested service"
            )
        if options is None:
            options = ScheduleOptions()
        if not isinstance(options, ScheduleOptions):
            raise TypeError(
                f"core Schedule options must be ScheduleOptions, got "
                f"{type(options).__name__}"
            )
        solved = select_atoms(build_schedule_tree(extract(root)), self._target, self.stage)
        return ScheduleResult(module=module, report=_project_report(root, solved, self.stage))


def _project_report(root: Function, solved: "TileGraph", stage: str) -> ScheduleReport:
    """Project one resource-solve result into the compact public report.

    A solve that did not prove optimality leaves no bound behind, so the only
    one this stage can state is the trivial zero.
    """
    makespan_ns = round(solved.decisions["makespan"] / _DURATION_SCALE)
    optimal = solved.decisions["status"] == "OPTIMAL"
    return ScheduleReport(
        root=root.name,
        target=root.target.name,
        stage=stage,
        status="OPTIMAL" if optimal else "FEASIBLE_NOT_PROVEN",
        objective_name="makespan",
        unit="ns",
        selected=makespan_ns,
        best_bound=makespan_ns if optimal else 0,
        gap=0.0 if optimal else 1.0,
    )


__all__ = ["_AmxCoreAnalysis", "_AmxCoreSchedule"]
