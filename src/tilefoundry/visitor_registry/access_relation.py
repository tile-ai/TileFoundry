"""Access relation analysis registry.

``AccessRelation = isl.multi_aff``. This module adds a per-op handler
registry so each HIR op can report the
input/output access relations at its current memory level (currently only
the GMEM (global) black-box level is implemented).


Usage::

    from tilefoundry.visitor_registry.access_relation import (
        register_access_relation,
        AccessRelations,
        OPAQUE,
    )

    @register_access_relation(MyOp)
    def _(call, ctx):
        return AccessRelations(inputs=(...,), outputs=(...,))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import isl

from .registries import AnalysisRegistry

# ─────────────────────────────────────────────────────────────────────
# OpaqueRelation — marker for "not affine-expressible at this level"
# ─────────────────────────────────────────────────────────────────────


class OpaqueRelation:
    """Marker object for an access relation that cannot be expressed in the
    affine framework at the queried memory level.

    Data-dependent or otherwise non-affine operations return ``OPAQUE`` for
    their boundaries because their access pattern is outside isl multi_aff /
    map.

    Distinct from ``isl.multi_aff`` / ``isl.map`` so downstream passes do not
    confuse "opaque" with "identity".
    """

    __slots__ = ()
    _instance: "OpaqueRelation | None" = None

    def __new__(cls) -> "OpaqueRelation":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "OPAQUE"

    def __reduce__(self):  # pragma: no cover - pickling round-trip
        return (OpaqueRelation, ())


OPAQUE = OpaqueRelation()


# Per spec ``tilegraph.md`` §3.4 the canonical carrier is ``isl.multi_aff``;
# ``isl.map`` is allowed when the relation is reduction-like or otherwise
# many-to-one.
AccessRelation = Union["isl.multi_aff", "isl.map", OpaqueRelation]


# ─────────────────────────────────────────────────────────────────────
# AccessRelations — per-call result
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    Mirrors ``TileGraph.input_access_relations`` / ``output_access_relations``
    semantics from ``docs/spec/tilegraph.md`` §3.7: one relation per boundary
    value, in boundary order.

    - ``inputs``: one entry per input arg of the Call (in argument order).
    - ``outputs``: one entry per output. Single-output ops have len 1;
      tuple-output ops have one entry per tuple field.
    """

    inputs: tuple[AccessRelation, ...]
    outputs: tuple[AccessRelation, ...]


# ─────────────────────────────────────────────────────────────────────
# AccessRelationResult — forward relation carrier (input-type driven)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccessRelationResult:
    """Forward access relation for one Call, built from input types alone.

    ``domain`` is the bounded iteration domain as an ``isl.set``: static dims
    are constant constraints, dynamic dims are isl parameters. ``maps`` holds
    one access ``isl.map`` per boundary value, in boundary order (inputs
    first, then outputs). ``param_map`` resolves each of ``domain``'s isl
    parameter names back to the ``ShapeDim`` it stands for; it is this
    Call's own data, never shared with any other Call's relation. The
    carrier holds no tensor shape — output shape is typeinfer-side data.
    """

    domain: "isl.set"
    maps: tuple["isl.map", ...]
    param_map: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────


access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")

# Forward relation registry: handlers build the relation from input types +
# op attributes only, so typeinfer can consume the relation without depending
# on the (not-yet-computed) output type.
type_relation_registry: AnalysisRegistry = AnalysisRegistry("type_relation")


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a GLOBAL-level access-relation handler.

    The handler signature is ``(call, ctx) -> AccessRelations``. Handlers may
    return ``isl.multi_aff`` / ``isl.map`` for affine-expressible boundaries
    or ``OPAQUE`` for boundaries that cannot be modelled at the queried level.
    """

    def decorator(fn: Callable) -> Callable:
        access_relation_registry.register(op_cls, fn)
        return fn

    return decorator


def register_type_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a forward type-relation builder.

    The handler signature is ``(call, input_types, ctx) -> AccessRelationResult``.
    It reads only ``input_types`` and the op's attributes — never the Call's own
    output type — so it can run before the output type exists.
    """

    def decorator(fn: Callable) -> Callable:
        type_relation_registry.register(op_cls, fn)
        return fn

    return decorator


def build_relation(call, input_types, ctx) -> "AccessRelationResult | None":
    """Build the forward access relation for *call*, or ``None`` if its op has
    no registered builder."""
    fn = type_relation_registry.lookup(type(call.target))
    if fn is None:
        return None
    return fn(call, input_types, ctx)


__all__ = [
    "OPAQUE",
    "OpaqueRelation",
    "AccessRelation",
    "AccessRelations",
    "AccessRelationResult",
    "access_relation_registry",
    "type_relation_registry",
    "register_access_relation",
    "register_type_relation",
    "build_relation",
]
