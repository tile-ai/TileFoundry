"""What a solved partition states about the program it was asked about.

The plan names the selection by the problem's own stable identities and states
what the solve proved about the objective. Nothing is rewritten: the Module the
caller passed in is the Module it gets back, and a selected placement is a
decision recorded next to the program rather than applied to it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule.plan import (
    PlanVerificationError,
    SchedulePlan,
    TargetSpecRef,
)

from .problem import PartitionProblem
from .solve import PartitionSolution


@dataclass(frozen=True)
class PartitionProof:
    """What the solve proved about its own objective.

    This is a result fact, not a summary of the run: the objective value, the
    bound the solver could establish, and whether the two met.
    """

    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_ns: int
    best_bound_ns: int
    proven_optimal: bool


@dataclass(frozen=True)
class SelectedOperation:
    """One selected candidate, and where in the topology and time it runs."""

    candidate_id: int
    site_id: int | None
    operation: str
    input_bucket_ids: tuple[int, ...]
    output_bucket_ids: tuple[int, ...]
    parallel_positions: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class SelectedPlacement:
    """One selected value, held in one selected type at one topology offset."""

    bucket_id: int
    value_id: int
    type_id: int
    offset: int


@dataclass(frozen=True)
class PartitionSchedulePlan(SchedulePlan):
    """The selection one partition solve committed to, and its proof."""

    topology: str
    target: TargetSpecRef
    placements: tuple[SelectedPlacement, ...]
    operations: tuple[SelectedOperation, ...]
    root_results: tuple[int, ...]
    proof: PartitionProof

    def verify(self, module: Module, function: Function, topology: Topology) -> None:
        """Check the plan against the request, without re-solving anything.

        Only what the plan itself claims is checked: that it answers the request
        it was made for, that every operation refers to placements the plan also
        carries, and that every root result was placed. Whether the schedule is
        good is what the proof states; whether it is well-formed is this.
        """
        if topology.name != self.topology:
            raise PlanVerificationError(
                f"partition plan decided topology {self.topology!r}, not "
                f"{topology.name!r}"
            )
        placed = {placement.bucket_id for placement in self.placements}
        placed_values = {placement.value_id for placement in self.placements}
        for operation in self.operations:
            for bucket_id in (*operation.input_bucket_ids, *operation.output_bucket_ids):
                if bucket_id not in placed:
                    raise PlanVerificationError(
                        f"partition plan operation {operation.candidate_id} refers to "
                        f"unplaced value bucket {bucket_id}"
                    )
            if operation.end_ns < operation.start_ns:
                raise PlanVerificationError(
                    f"partition plan operation {operation.candidate_id} ends before "
                    "it starts"
                )
            if operation.parallel_positions < 0:
                raise PlanVerificationError(
                    f"partition plan operation {operation.candidate_id} occupies "
                    f"{operation.parallel_positions} parallel positions"
                )
        for value_id in self.root_results:
            if value_id not in placed_values:
                raise PlanVerificationError(
                    f"partition plan leaves root result value {value_id} unplaced"
                )
        if self.proof.best_bound_ns > self.proof.objective_ns:
            raise PlanVerificationError(
                "partition plan states a bound above its own objective"
            )

    def to_json(self) -> str:
        """Render the whole selection as sorted-key JSON."""
        return json.dumps(asdict(self), sort_keys=True)

    def render(self) -> str:
        """Render the selection as one line per operation, in solve order."""
        lines = [
            f"partition {self.topology} on {self.target.device_id} "
            f"({self.proof.status}, makespan {self.proof.objective_ns}ns)"
        ]
        for operation in self.operations:
            lines.append(
                f"  {operation.operation} x{operation.parallel_positions} "
                f"[{operation.start_ns}, {operation.end_ns})"
            )
        return "\n".join(lines)


def export_partition_plan(
    problem: PartitionProblem, solution: PartitionSolution
) -> PartitionSchedulePlan:
    """State the solved selection in the plan's own stable vocabulary."""
    intervals = dict(solution.candidate_intervals_ns)
    offsets = dict(solution.bucket_offsets)
    operations = tuple(
        SelectedOperation(
            candidate_id=candidate_id,
            site_id=problem.candidates[candidate_id].site_id,
            operation=type(problem.candidates[candidate_id].op).__name__,
            input_bucket_ids=problem.candidates[candidate_id].input_bucket_ids,
            output_bucket_ids=problem.candidates[candidate_id].output_bucket_ids,
            parallel_positions=problem.candidates[candidate_id].topology_count,
            start_ns=intervals[candidate_id].start_ns if candidate_id in intervals else 0,
            end_ns=intervals[candidate_id].end_ns if candidate_id in intervals else 0,
        )
        for candidate_id in solution.selected_candidate_ids
    )
    placements = tuple(
        SelectedPlacement(
            bucket_id=bucket_id,
            value_id=problem.buckets[bucket_id].value_id,
            type_id=problem.buckets[bucket_id].type_id,
            offset=offsets.get(bucket_id, 0),
        )
        for bucket_id in solution.selected_bucket_ids
    )
    return PartitionSchedulePlan(
        topology=problem.topology.name,
        target=problem.facts.spec,
        placements=placements,
        operations=operations,
        root_results=problem.root_value_ids,
        proof=PartitionProof(
            status=solution.status,
            objective_ns=solution.makespan_ns,
            best_bound_ns=solution.best_bound_ns,
            proven_optimal=solution.status == "OPTIMAL",
        ),
    )


__all__ = [
    "PartitionProof",
    "PartitionSchedulePlan",
    "SelectedOperation",
    "SelectedPlacement",
    "export_partition_plan",
]
