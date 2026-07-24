"""``kernelize`` -- the agent-friendly compiler path's polyhedral stage:
``extract(HIR Function) -> TileGraph`` then ``schedule(TileGraph) ->
ScheduleTree`` (isl schedule tree). Lives alongside the existing HIR ->
TIR -> CUDA path (``compile.py`` / ``passes`` / ``codegen`` / ``ir``),
never on it: nothing here is imported by, or mutates, that path.
"""
from __future__ import annotations

from .emit_scaffold import (
    EmitScaffoldError,
    HoleContract,
    Skeleton,
    Swimlane,
    TensorView,
    emit_scaffold,
)
from .extract import DEFAULT_TILE_SIZE, ExtractError, extract
from .schedule_tree import ScheduleTree, schedule
from .tile_graph import TileGraph, TileUnit

__all__ = [
    "TileGraph",
    "TileUnit",
    "extract",
    "ExtractError",
    "DEFAULT_TILE_SIZE",
    "ScheduleTree",
    "schedule",
    "Skeleton",
    "Swimlane",
    "TensorView",
    "HoleContract",
    "EmitScaffoldError",
    "emit_scaffold",
]
