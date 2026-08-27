"""Movement projection entry points.

M8 removes the family-local movement accumulators. Scope/Access supplies the
replacement in M9; these names remain as loud failures while callers migrate.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.types import TensorType, Type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    leaves_of,
    reached_elements,
    reached_leaves,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import CostContext

from .errors import AnalysisError
from .metadata import TrafficMetadata

_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


def _bytes_for(held: Type, elements: int | None) -> int | None:
    """The bytes *elements* of *held* occupy."""
    if elements is None or not isinstance(held, TensorType):
        return None
    bits = getattr(held.dtype, "bit_width", None)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        return None
    return -(-(elements * bits) // 8)


def _reached_bytes(
    boundaries: tuple[tuple[Type, object], ...], umat_level: str | None
) -> tuple[int, dict[str, int]] | None:
    """Count bytes reached by boundaries without building family projections."""
    total = 0
    levels: dict[str, int] = {}
    for held, pattern in boundaries:
        leaves = leaves_of(held)
        if not leaves:
            return None
        if len(leaves) == 1:
            taken = {0: _bytes_for(leaves[0], reached_elements(pattern))}
        else:
            reached = reached_leaves(pattern, len(leaves))
            if reached is None:
                return None
            taken = {index: static_bytes(leaves[index]) for index in sorted(reached)}
        for index, size in taken.items():
            if size is None:
                return None
            total += size
            leaf = leaves[index]
            level = umat_level if leaf.storage is StorageKind.UMAT else str(leaf.storage)
            levels[level] = levels.get(level, 0) + size
    return total, levels


def call_traffic(
    expr: Call, whole: CostContext, local: CostContext
) -> TrafficMetadata:
    """Fail until M9 supplies movement from shared Access records."""
    raise AnalysisError(
        "movement projection was removed in M8; M9 rebuilds it from Scope/Access"
    )


def add_traffic(*_args: object, **_kwargs: object) -> None:
    """Fail until M9 supplies movement aggregation from shared Access records."""
    raise AnalysisError(
        "movement aggregation was removed in M8; M9 rebuilds it from Scope/Access"
    )


__all__ = ["add_traffic", "call_traffic"]
