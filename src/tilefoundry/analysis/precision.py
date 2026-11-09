"""Directional guarantees shared by analysis quantities."""

from enum import Enum


class Precision(Enum):
    """Describe a quantity relative to its true value, preserving direction."""

    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    UNKNOWN = "unknown"

    def join(self, other: "Precision") -> "Precision":
        """Combine evidence without losing opposing or unknown directions."""
        if self is Precision.EXACT:
            return other
        if other is Precision.EXACT or self is other:
            return self
        return Precision.UNKNOWN


__all__ = ["Precision"]
