"""Validation shared by Target-owned Facts projections."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

FactsT = TypeVar("FactsT")


class TargetFactsError(Exception):
    """A Target failed to provide the requested immutable Facts aggregate."""


def facts_result(
    target: object, facts_type: type[FactsT], value: object
) -> FactsT:
    """Validate and return one Facts value provided by *target*."""
    if not isinstance(facts_type, type):
        raise TargetFactsError(
            f"{type(target).__name__}: Facts type must be a class, got "
            f"{type(facts_type).__name__}"
        )
    if not dataclasses.is_dataclass(facts_type) or not facts_type.__dataclass_params__.frozen:
        raise TargetFactsError(
            f"{type(target).__name__}: {facts_type.__name__} must be a frozen "
            "dataclass Facts aggregate"
        )
    if not isinstance(value, facts_type):
        raise TargetFactsError(
            f"{type(target).__name__}: Facts projection for {facts_type.__name__} "
            f"returned {type(value).__name__}"
        )
    return value


__all__ = ["FactsT", "TargetFactsError", "facts_result"]
