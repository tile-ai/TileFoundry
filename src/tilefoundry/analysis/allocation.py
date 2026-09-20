"""Internal values and solver results for physical memory placement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr

from .metadata import ValueLifetime


@dataclass(frozen=True)
class AllocationValue:
    """Connect one logical expression to its target-aware lifetime."""

    value: Expr
    lifetime: ValueLifetime


__all__ = ["AllocationValue"]
