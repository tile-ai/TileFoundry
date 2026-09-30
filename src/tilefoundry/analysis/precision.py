"""Directional guarantees shared by analysis quantities."""

from enum import Enum


class AnalysisPrecision(Enum):
    """Describe a quantity relative to its true value, preserving direction."""

    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    UNKNOWN = "unknown"

    def join(self, other: "AnalysisPrecision") -> "AnalysisPrecision":
        """Combine evidence without losing opposing or unknown directions."""
        if self is AnalysisPrecision.EXACT:
            return other
        if other is AnalysisPrecision.EXACT or self is other:
            return self
        return AnalysisPrecision.UNKNOWN
