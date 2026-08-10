"""Register per-operation affine access relations.

Handlers return input and output isl relations from call types and attributes.
``OPAQUE`` marks boundaries that cannot be expressed at the queried memory
level. The GMEM black-box level is currently supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import isl

from .registries import AnalysisRegistry


class OpaqueRelation:
    """Represent OpaqueRelation.

    Marker object for an access relation that cannot be expressed in the
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




AccessRelation = Union["isl.multi_aff", "isl.map", OpaqueRelation]







@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    One relation per boundary value, in boundary order.

    - ``inputs``: one entry per input arg of the Call (in argument order).
    - ``outputs``: one entry per output. Single-output ops have len 1;
      tuple-output ops have one entry per tuple field.
    """

    inputs: tuple[AccessRelation, ...]
    outputs: tuple[AccessRelation, ...]







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







access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")




type_relation_registry: AnalysisRegistry = AnalysisRegistry("type_relation")


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a GLOBAL-level access-relation handler.

    The handler signature is ``(call, ctx) -> AccessRelations``. Handlers may
    return ``isl.multi_aff`` / ``isl.map`` for affine-expressible boundaries
    or ``OPAQUE`` for boundaries that cannot be modelled at the queried level.
    """
    return access_relation_registry.decorator()(op_cls)


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


def identity_relations(n_inputs: int) -> Callable[..., AccessRelations]:
    """Identity relations.

    Factory for a GLOBAL-level access-relation handler whose ``n_inputs``
    inputs and single output are all elementwise identity.

    Each input contributes its own-rank identity; the output uses its own
    rank. A structural (non-tensor) input arg — e.g. ``TupleGetItem``'s tuple
    operand — has no shape of its own, so it borrows the output's rank.
    """

    def _handler(call, ctx) -> AccessRelations:
        out_rank = len(ctx.type_of(call).shape)

        def _rank_of(arg) -> int:
            ty = ctx.type_of(arg)
            return len(ty.shape) if hasattr(ty, "shape") else out_rank

        inputs = tuple(_identity(_rank_of(call.args[i])) for i in range(n_inputs))
        return AccessRelations(inputs=inputs, outputs=(_identity(out_rank),))

    return _handler


def register_type_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a forward type-relation builder.

    The handler signature is ``(call, input_types, ctx) -> AccessRelationResult``.
    It reads only ``input_types`` and the op's attributes — never the Call's own
    output type — so it can run before the output type exists.
    """
    return type_relation_registry.decorator()(op_cls)


def build_relation(call, input_types, ctx) -> "AccessRelationResult | None":
    """Build relation.

    Build the forward access relation for *call*, or ``None`` if its op has
    no registered builder.
    """
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
    "identity_relations",
    "build_relation",
]
