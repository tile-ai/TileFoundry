"""``kernelize`` -- the agent-friendly compiler path's polyhedral stage:
``extract(HIR Function) -> TileGraph`` then ``schedule(TileGraph) ->
TileGraph`` (isl schedule tree filled in). Lives alongside the existing
HIR -> TIR -> CUDA path (``compile.py`` / ``passes`` / ``codegen`` /
``ir``), never on it: nothing here is imported by, or mutates, that path.
"""
from __future__ import annotations

from .emit_scaffold import (
    BufferAccess,
    EmitScaffoldError,
    HoleContract,
    Skeleton,
    Swimlane,
    emit_scaffold,
)
from .extract import ExtractError, extract
from .schedule_tree import schedule
from .solve_resources import SolveResourcesError, solve_resources
from .target_facts import AtomFact, candidate_atoms
from .tile_graph import TileGraph, TileUnit

__all__ = [
    "TileGraph",
    "TileUnit",
    "extract",
    "ExtractError",
    "schedule",
    "Skeleton",
    "Swimlane",
    "BufferAccess",
    "HoleContract",
    "EmitScaffoldError",
    "emit_scaffold",
    "AtomFact",
    "candidate_atoms",
    "SolveResourcesError",
    "solve_resources",
]
