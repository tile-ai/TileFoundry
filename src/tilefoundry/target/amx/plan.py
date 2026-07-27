"""What the AMX core solve decides.

This level chooses instructions and resources over the schedule tree without
rewriting the program, so the plan is the objective the solve reached and the
level it reached it at. There is no materialized Module here, because nothing was
materialized -- reporting one would claim a rewrite that did not happen.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.schedule import ScheduleReport
from tilefoundry.schedule.plan import PlanVerificationError, SchedulePlan


@dataclass(frozen=True)
class CoreSchedulePlan(SchedulePlan):
    """What the core-level solve proved about the program it was given."""

    report: ScheduleReport

    def verify(self, module, function, topology) -> None:
        """Check that the reported objective answers the request it was made for."""
        if self.report.root != function.name:
            raise PlanVerificationError(
                f"core plan reports root {self.report.root!r} but was asked for "
                f"{function.name!r}"
            )
        if self.report.topology != topology.name:
            raise PlanVerificationError(
                f"core plan reports topology {self.report.topology!r} but was "
                f"asked for {topology.name!r}"
            )
        if self.report.gap < 0.0:
            raise PlanVerificationError(
                f"core plan reports a negative optimality gap {self.report.gap}"
            )

    def to_json(self) -> str:
        """The objective summary, which is the whole plan."""
        return self.report.to_json()

    def render(self) -> str:
        """The objective summary, which is the whole plan."""
        return self.report.to_markdown()


__all__ = ["CoreSchedulePlan"]
