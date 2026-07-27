"""The spatial partition scheduling algorithm family.

The names here are the stages one registered algorithm composes its solve from,
in the order it composes them: extract what the program states, ask the hardware
once, close the problem, solve it, export the plan. They are this family's own
vocabulary and no other algorithm reads them.
"""

from __future__ import annotations

from .facts import PartitionFacts, PartitionFactsError, PartitionFactsQuery
from .plan import (
    PartitionedOperation,
    PartitionProof,
    PartitionSchedulePlan,
    PlacedValue,
    PositionInterval,
    TimeInterval,
    export_partition_plan,
)
from .problem import PartitionProblem, PartitionProblemError, build_partition_problem
from .program import PartitionProgram, PartitionProgramError, build_partition_program
from .solve import PartitionSolution, PartitionSolveError, solve_partition_problem

__all__ = [
    "PartitionFacts",
    "PartitionFactsError",
    "PartitionFactsQuery",
    "PartitionProblem",
    "PartitionProblemError",
    "PartitionProgram",
    "PartitionProgramError",
    "PartitionProof",
    "PartitionSchedulePlan",
    "PartitionedOperation",
    "PlacedValue",
    "PositionInterval",
    "TimeInterval",
    "PartitionSolution",
    "PartitionSolveError",
    "build_partition_problem",
    "build_partition_program",
    "export_partition_plan",
    "solve_partition_problem",
]
