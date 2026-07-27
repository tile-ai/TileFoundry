"""Typed target facts consumed by the private partition scheduler.

These are every number the partition problem is closed with. Once they are
projected, neither the problem nor the solve holds a Target: what the hardware
contributes to a decision is exactly this record, and it is readable in one
place instead of inferred from the call sites that used to reach for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.schedule.plan import TargetSpecRef


class PartitionFactsError(ValueError):
    """A projected fact the partition algorithm needs is absent or unusable."""


@dataclass(frozen=True)
class PartitionFactsQuery:
    """The one topology level a projection is asked to describe."""

    topology: str


@dataclass(frozen=True)
class PartitionFacts:
    """All concrete hardware information required to close one partition."""

    topology: str
    spec: TargetSpecRef
    parallel_units: int
    memory_bandwidth_bytes_per_second: int
    memory_capacity_bytes: int
    peak_flops_per_second: tuple[tuple[DType, int], ...]

    def peak_flops(self, dtype: DType) -> int:
        """The dense peak rate stated for *dtype*.

        A dtype the hardware documents no rate for is an error rather than a
        zero or a substituted neighbour: costing work at a rate nobody published
        would put a number in the plan that no document supports.
        """
        for candidate, value in self.peak_flops_per_second:
            if candidate == dtype:
                return value
        stated = ", ".join(sorted(item[0].name for item in self.peak_flops_per_second))
        raise PartitionFactsError(
            f"{self.spec.device_id} states no dense peak rate for {dtype.name}; "
            f"it states {stated or 'none'}"
        )


__all__ = ["PartitionFacts", "PartitionFactsError", "PartitionFactsQuery"]
