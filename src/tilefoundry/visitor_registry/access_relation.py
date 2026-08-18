"""Register what one operation reads, writes, and where its result's bytes live.

Boundary handlers return input and output isl relations from call types and
attributes. ``OPAQUE`` marks boundaries that cannot be expressed at the queried
memory level. The GMEM black-box level is currently supported.

The same registration says whether the result owns its bytes or reuses an
operand's. An Op states that as a claim about its own types and operands; the
whole-function proof that settles it belongs to whoever walks the function. An
identity relation alone claims nothing, so an Op that says nothing produces.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Union

import isl

from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of

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
    - ``storage``: where the result's bytes live, or ``None`` for an Op that
      produces its own. Reading an operand at the same indices is not the same
      as being that operand, so an identity relation claims nothing by itself.
    """

    inputs: tuple[AccessRelation, ...]
    outputs: tuple[AccessRelation, ...]
    storage: "StorageClaim | None" = None







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







class StorageEffect(enum.Enum):
    """What one Op claims about where its result's bytes live."""

    PRODUCE = "produce"
    FORWARD = "forward"
    UPDATE = "update"


@dataclass(frozen=True)
class StorageSpan:
    """One run of bytes inside one operand's own buffer."""

    operand: int
    offset: int
    size: int


@dataclass(frozen=True)
class StorageClaim:
    """Which operands a result lives in, and where inside them if that is known.

    ``spans_required`` marks a claim that is only true because of where the
    pieces sit -- putting several of them side by side, say. Without it, a claim
    whose spans do not resolve still stands on its operands alone, which is what
    a re-indexing of one operand is whether or not its address is known.
    """

    effect: StorageEffect
    operands: tuple[int, ...]
    spans: tuple[StorageSpan, ...] = ()
    spans_required: bool = False


access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")




type_relation_registry: AnalysisRegistry = AnalysisRegistry("type_relation")


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a GLOBAL-level access-relation handler.

    The handler signature is ``(call, ctx) -> AccessRelations``. Handlers may
    return ``isl.multi_aff`` / ``isl.map`` for affine-expressible boundaries or
    ``OPAQUE`` for boundaries that cannot be modelled at the queried level, and
    state on the same value where the result's bytes live.
    """
    return access_relation_registry.decorator()(op_cls)


def declared_storage(call, ctx) -> "StorageClaim | None":
    """What one Call's Op claims about where its result's bytes live.

    Asking the Op means building its relations, because one registration answers
    both questions. Anything it cannot answer produces: over-reporting one
    allocation is safe and missing one is not.
    """
    handler = access_relation_registry.lookup(type(call.target))
    if handler is None:
        return None
    try:
        return handler(call, ctx).storage
    except (TypeError, ValueError, KeyError, IndexError, NotImplementedError):
        return None


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


def identity_relations(
    n_inputs: int, storage: "Callable[..., StorageClaim | None] | None" = None
) -> Callable[..., AccessRelations]:
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
        return AccessRelations(
            inputs=inputs,
            outputs=(_identity(out_rank),),
            storage=None if storage is None else storage(call, ctx),
        )

    return _handler


def static_bytes(type_: "Type") -> int | None:
    """How many bytes a Type holds, or ``None`` when it is not static."""
    if isinstance(type_, TupleType):
        sizes = [static_bytes(field_) for field_ in type_.fields]
        if any(size is None for size in sizes):
            return None
        return sum(size for size in sizes if size is not None)
    if not isinstance(type_, TensorType):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in type_.shape):
        return None
    try:
        amount = tensor_bytes(type_)
    except (TypeError, ValueError):
        return None
    return amount if isinstance(amount, int) else None


def dense(type_: "Type") -> bool:
    """Whether a Type's own elements sit in one unbroken row-major run.

    A sharded Type is read through its per-position tile, which is the run one
    position addresses. Anything this cannot decide -- a composed layout, a
    symbolic stride -- is not dense here, so the proof that needed it fails.
    """
    if not isinstance(type_, TensorType):
        return False
    layout = type_.layout
    shard = shard_layout_of(layout)
    if shard is not None:
        layout = shard.layout
    if layout is None:
        return True
    if not isinstance(layout, Layout):
        return False
    if layout.strides is None:
        return True
    expected = try_c_order_strides(flatten(layout.shape))
    return expected is not None and tuple(layout.strides) == expected


def same_placement(left: "Type", right: "Type") -> bool:
    """Whether two Types name the same storage, element size, and positions."""
    if not (isinstance(left, TensorType) and isinstance(right, TensorType)):
        return False
    if left.storage != right.storage or left.dtype != right.dtype:
        return False
    left_shard, right_shard = shard_layout_of(left.layout), shard_layout_of(right.layout)
    if left_shard is None or right_shard is None:
        return left_shard is None and right_shard is None
    return left_shard.mesh == right_shard.mesh


def forward_whole(call, operand: int, ctx) -> StorageClaim:
    """Forward all of *operand*, with its address when the sizes are static."""
    size = static_bytes(ctx.type_of(call))
    if size is None or size != static_bytes(ctx.type_of(call.args[operand])):
        return StorageClaim(StorageEffect.FORWARD, (operand,))
    return StorageClaim(
        StorageEffect.FORWARD, (operand,), (StorageSpan(operand, 0, size),)
    )


def update_destination(call, ctx, *, destination: int) -> "StorageClaim | None":
    """Claim that the result is the destination's buffer after an overwrite.

    Whether overwriting is free is not a question about this Call: it depends on
    who else still reads those bytes and on whose buffer it is. All this states
    is that the result is that operand, laid out the same way and the same size.
    """
    if not same_placement(ctx.type_of(call.args[destination]), ctx.type_of(call)):
        return None
    size = static_bytes(ctx.type_of(call))
    if size is None or size != static_bytes(ctx.type_of(call.args[destination])):
        return None
    return StorageClaim(
        StorageEffect.UPDATE, (destination,), (StorageSpan(destination, 0, size),)
    )


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
    "StorageClaim",
    "StorageEffect",
    "StorageSpan",
    "declared_storage",
    "dense",
    "forward_whole",
    "same_placement",
    "static_bytes",
    "update_destination",
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
