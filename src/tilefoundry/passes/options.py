"""The options every pass is created with."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    """Whether the pipeline lowers HIR to TIR or leaves the program as authored."""

    lower: bool = True


__all__ = ["Options"]
