"""What the CUDA CTA solve decides.

The decision this solve makes is a rewritten program: it places work across the
CTA hierarchy by splitting the entry function into per-CTA functions. So the
materialized Module is the plan's content, not a side effect of producing it --
there is nothing else to hand an agent that would say what was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.schedule import ScheduleReport
from tilefoundry.schedule.plan import PlanVerificationError, SchedulePlan


@dataclass(frozen=True)
class CtaSchedulePlan(SchedulePlan):
    """A materialized per-CTA program and what the solve proved about it."""

    report: ScheduleReport
    materialized: Module

    def verify(self, module: Module, function, topology) -> None:
        """Check the plan against the request it answers.

        Only references are checked. Whether the materialized program is
        internally well formed is settled by IR verification while the plan is
        built, and how good the schedule is was settled by the solve; neither is
        re-decided here.
        """
        if self.report.root != function.name:
            raise PlanVerificationError(
                f"cta plan reports root {self.report.root!r} but was asked for "
                f"{function.name!r}"
            )
        if self.report.topology != topology.name:
            raise PlanVerificationError(
                f"cta plan reports topology {self.report.topology!r} but was "
                f"asked for {topology.name!r}"
            )
        if self.materialized.entry != module.entry:
            raise PlanVerificationError(
                f"cta plan materialized entry {self.materialized.entry!r}, "
                f"which is not the requested module's entry {module.entry!r}"
            )
        if self.report.gap < 0.0:
            raise PlanVerificationError(
                f"cta plan reports a negative optimality gap {self.report.gap}"
            )

    def to_json(self) -> str:
        """The objective summary. The program itself is rendered, not encoded."""
        return self.report.to_json()

    def render(self) -> str:
        """The objective summary above the materialized program as source."""
        return f"{self.report.to_markdown()}\n\n{as_script(self.materialized)}"


__all__ = ["CtaSchedulePlan"]
