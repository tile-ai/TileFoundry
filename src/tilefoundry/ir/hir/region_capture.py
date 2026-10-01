"""Shared invariant captures of structured HIR regions."""

from __future__ import annotations

from tilefoundry.ir.core import Expr, Var

from .loop_region import LoopRegion
from .mesh_region import MeshRegion

CapturingRegion = MeshRegion | LoopRegion


def region_captures(region: CapturingRegion) -> tuple[tuple[Var, Expr], ...]:
    """Pair invariant body parameters with values evaluated outside the region."""
    carried = len(region.yield_values) if isinstance(region, LoopRegion) else 0
    return tuple(zip(region.params[carried:], region.args[carried:], strict=True))


__all__ = ["CapturingRegion", "region_captures"]
