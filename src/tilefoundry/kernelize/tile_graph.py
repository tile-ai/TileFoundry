"""``TileGraph`` / ``TileUnit`` — the polyhedral model plus its schedule.

``docs/spec/tilegraph.md`` defines ``TileGraph`` conceptually as a
per-level SSA DAG with a single ``domain: isl.set`` and a
``body: DiGraph[ITileNode]``. That single-domain shape does not carry a
*multi-statement* polyhedral program: feeding ``isl.schedule_constraints``
needs one **union** domain (one named tuple per statement) plus the
access relations and dependences over that union. This module is
therefore a from-scratch V1 shape sized for the isl scheduler, not an
implementation of the spec's node types (``ITileNode`` / ``DiGraph`` /
``AccessRelation`` are pass-private here, not reused).

``domain``/``reads``/``writes`` are per-statement pieces unioned
together, one tuple name (``TileUnit.name``) per statement. ``deps`` is
auto-inferred by :func:`extract.extract` from ``reads``/``writes`` via
``isl.union_access_info(...).compute_flow()``. ``tree``/``ring``/
``decisions`` start empty from ``extract`` and fill in as the same object
flows through ``schedule()`` then ``solve_resources()`` -- one object
carries the whole pipeline's state, so every later stage (``schedule``,
``solve_resources``, ``emit_scaffold``) takes and returns a single
``TileGraph``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import isl


@dataclass(frozen=True)
class TileUnit:
    """One statement's identity.

    ``name`` is the isl tuple name shared by this statement's pieces of
    ``TileGraph.domain``/``reads``/``writes``/``deps`` (e.g. ``"MM"``).
    ``op`` is the HIR ``Call`` (op@site) that produced this statement --
    the call node itself, not just its bare ``Op``, so a consumer can
    still recover ``op.target`` / ``op.args`` / ``op.type``.
    """

    name: str
    op: object


@dataclass(frozen=True)
class TileGraph:
    """Polyhedral extraction of one HIR ``Function`` body, plus its schedule.

    ``domain``/``reads``/``writes`` are unions of per-``TileUnit`` pieces,
    each tuple-named by its producing statement (domain/reads/writes) or
    accessed buffer (reads/writes range). ``deps`` is the auto-inferred
    RAW must-dependence relation between statement instances (see
    ``extract.py``). ``params`` resolves any dynamic-shape isl parameter
    name appearing in ``domain`` back to its ``ShapeDim`` (empty for an
    all-static V1 extraction).

    ``tree`` is the isl schedule computed by ``schedule()`` (``None``
    before that runs). ``ring`` maps a buffer name to its software-
    pipelining depth; ``decisions`` carries ``solve_resources()``'s full
    per-statement resource picks plus its solve status/makespan. Both are
    plain fields, never an isl mark payload -- isl marks are process-
    global C state, not a place for an arbitrary Python object to live.
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    tree: "isl.schedule | None" = None
    ring: dict = field(default_factory=dict)
    decisions: dict | None = None


__all__ = ["TileGraph", "TileUnit"]
