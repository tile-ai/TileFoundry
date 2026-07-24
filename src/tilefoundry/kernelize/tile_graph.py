"""``TileGraph`` / ``TileUnit`` — the extraction-side polyhedral IR.

``docs/spec/tilegraph.md`` defines ``TileGraph`` conceptually as a
per-level SSA DAG with a single ``domain: isl.set`` and a
``body: DiGraph[ITileNode]``. That single-domain shape does not carry a
*multi-statement* polyhedral program: feeding ``isl.schedule_constraints``
needs one **union** domain (one named tuple per statement) plus the
access relations and dependences over that union. This module is
therefore a from-scratch V1 shape sized for the isl scheduler, not an
implementation of the spec's node types (``ITileNode`` / ``DiGraph`` /
``AccessRelation`` are pass-private here, not reused).

Field/type provenance:

- ``domain``/``reads``/``writes`` are per-statement pieces unioned
  together, one tuple name (``TileUnit.name``) per statement.
- ``deps`` is not authored -- it is derived by
  :func:`tilefoundry.kernelize.extract.extract` from ``reads``/``writes``
  via ``isl.union_access_info(...).compute_flow()`` (see that module for
  the algorithm), mirroring the validated
  ``m1_deps_probe.py`` technique.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    """Polyhedral extraction of one HIR ``Function`` body.

    ``domain``/``reads``/``writes`` are unions of per-``TileUnit`` pieces,
    each tuple-named by its producing statement (domain/reads/writes) or
    accessed buffer (reads/writes range). ``deps`` is the auto-inferred
    RAW must-dependence relation between statement instances (see
    ``extract.py``). ``params`` resolves any dynamic-shape isl parameter
    name appearing in ``domain`` back to its ``ShapeDim`` (empty for an
    all-static V1 extraction).
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict


__all__ = ["TileGraph", "TileUnit"]
