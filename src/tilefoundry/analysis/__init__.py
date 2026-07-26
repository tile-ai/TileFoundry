"""Direct public Analysis fact contract."""

from __future__ import annotations

from typing import Protocol

from .analyzer import AnalysisError, AnalysisOptions, AnalysisResult, analyze
from .atom_facts import AtomFact
from .metadata import (
    FootprintMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)
from .poly import ExtractError, TileGraph, TileUnit, extract


class Analysis(Protocol):
    """One stage's target-dependent facts, for the Schedule stage that
    solves over them: the polyhedral model itself is target-independent
    (:func:`extract`), the atom catalogue is not."""

    stage: str

    def candidate_atoms(self, op: "Call") -> list[AtomFact]: ...


__all__ = [
    "Analysis",
    "AnalysisError",
    "AnalysisOptions",
    "AnalysisResult",
    "AtomFact",
    "ExtractError",
    "FootprintMetadata",
    "RooflineMetadata",
    "TileGraph",
    "TileUnit",
    "TimelineMetadata",
    "TrafficBytes",
    "analyze",
    "extract",
]
