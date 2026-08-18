"""Register what one operation reads, writes, and where its result's bytes live.

Boundary handlers return one relation per boundary value: an isl relation where
the addresses are affine, an ``IndexedAccess`` or ``WindowAccess`` where a
runtime value decides them. All of them say how much is read, and there is no
way to say nothing.

The same registration says whether the result owns its bytes or reuses an
operand's, as a claim about that Op's own types. Whoever walks the function
settles it; an identity relation alone claims nothing.
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


@dataclass(frozen=True)
class IndexedAccess:
    """One boundary read through the values of another operand.

    A gather names which source coordinate it wants by the element it reads out
    of ``index_operand``, so the address is not known here. How much it reads is:
    one slice of the source per index element. That is what a cost or a
    footprint asks for, so this states the pattern rather than giving up on it.
    """

    index_operand: int
    source_axis: int


@dataclass(frozen=True)
class OperandValue:
    """One runtime number an access depends on, and what it may be.

    ``element`` picks a field when the operand is a tuple of scalars. ``bound``
    is the Op's own contract for the value, which is what lets a quantity stay a
    checkable range instead of widening to the whole operand.
    """

    operand: int
    element: int | None = None
    bound: tuple[int, int] | None = None


@dataclass(frozen=True)
class WindowAccess:
    """One boundary read or written as a window, one entry per axis.

    An ``offset`` says where the window starts on its axis and an ``extent`` how
    far it runs; either may be a number written down or an ``OperandValue`` only
    known at run time. Where the window sits never changes how much it covers,
    which is why an unbound offset costs a quantity nothing. ``complement``
    marks the part of the container the window leaves alone -- the same question
    answered from the other side.
    """

    offsets: tuple["OperandValue | int", ...]
    extents: tuple["OperandValue | object", ...]
    complement: bool = False


AccessRelation = Union["isl.multi_aff", "isl.map", IndexedAccess, WindowAccess]







@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    One relation per boundary value, in boundary order.

    - ``inputs``: one entry per input arg of the Call (in argument order).
    - ``outputs``: one entry per output. Single-output ops have len 1;
      tuple-output ops have one entry per tuple field.
    - ``storage_effect``: where the result's bytes live. An Op that says nothing
      produces its own, because reading an operand at the same indices is not
      the same as being that operand.
    """

    inputs: tuple[AccessRelation, ...]
    outputs: tuple[AccessRelation, ...]
    storage_effect: "StorageEffectClaim | None" = None







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







class StorageEffectKind(enum.Enum):
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
class StorageEffectClaim:
    """Which operands a result lives in, and where inside them if that is known.

    ``spans_required`` marks a claim that is only true because of where the
    pieces sit -- putting several of them side by side, say. Without it, a claim
    whose spans do not resolve still stands on its operands alone, which is what
    a re-indexing of one operand is whether or not its address is known.
    """

    kind: StorageEffectKind = StorageEffectKind.PRODUCE
    operands: tuple[int, ...] = ()
    spans: tuple[StorageSpan, ...] = ()
    spans_required: bool = False


access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")




type_relation_registry: AnalysisRegistry = AnalysisRegistry("type_relation")


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a GLOBAL-level access-relation handler.

    The handler signature is ``(call, ctx) -> AccessRelations``. Every boundary
    gets a relation: ``isl.multi_aff`` / ``isl.map`` where the addresses are
    affine, ``IndexedAccess`` or ``WindowAccess`` where a runtime value decides
    them. The same value states where the result's bytes live.
    """
    return access_relation_registry.decorator()(op_cls)


def declared_storage(call, ctx) -> "StorageEffectClaim | None":
    """What one Call's Op claims about where its result's bytes live.

    Asking the Op means building its relations, because one registration answers
    both questions. A handler that cannot build them says so and is heard: the
    fail-closed direction belongs to the proof, not to the claim, or a broken
    handler would read as an Op that merely produces.
    """
    handler = access_relation_registry.lookup(type(call.target))
    return None if handler is None else handler(call, ctx).storage_effect


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


def identity_relations(
    n_inputs: int, storage: "Callable[..., StorageEffectClaim | None] | None" = None
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
            storage_effect=None if storage is None else storage(call, ctx),
        )

    return _handler


@dataclass(frozen=True)
class AccessQuantity:
    """How many elements one boundary touches, and how well that is known.

    ``lower`` and ``upper`` are equal whenever the count does not depend on a
    value nobody has bound yet. When they differ the range comes from the Op's
    own contract, which is a smaller thing to say than "all of it" and a
    checkable one: a prediction takes ``upper`` and says that it did.
    """

    lower: int
    upper: int

    @property
    def exact(self) -> bool:
        """Whether one number answers this, rather than a range."""
        return self.lower == self.upper


def _elements(shape: tuple) -> int | None:
    """How many elements a shape holds, or ``None`` when it is not concrete."""
    total = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            return None
        total *= extent
    return total


def _span(value, ctx, call) -> tuple[int, int] | None:
    """One axis extent as the range it may take, or ``None`` when unknowable."""
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, OperandValue):
        return value.bound
    return None


def _window_span(relation: WindowAccess, container: tuple, ctx, call):
    """How many elements the window covers, as a range over its axes."""
    if len(relation.extents) != len(container):
        return None
    low = high = 1
    for axis, extent in enumerate(relation.extents):
        span = _span(extent, ctx, call)
        if span is None:
            return None
        limit = container[axis]
        if not isinstance(limit, int) or isinstance(limit, bool):
            return None
        low *= min(span[0], limit)
        high *= min(span[1], limit)
    return low, high


def access_elements(
    relations: AccessRelations, call, ctx, *, boundary: int, output: bool = False
) -> AccessQuantity | None:
    """How many elements one boundary of *call* touches in one execution.

    Boundaries are numbered within their side: ``boundary`` indexes ``inputs``
    unless ``output`` is set. Every kind of relation answers this, which is the
    point of not having a way to say nothing. An affine boundary is counted from
    the relation itself over the result's iteration domain, a gather reads one
    source slice per index element, and a window reads its own extent -- or, for
    a complement, everything the window leaves.
    """
    chosen = relations.outputs if output else relations.inputs
    if not 0 <= boundary < len(chosen):
        return None
    relation = chosen[boundary]
    if isinstance(relation, IndexedAccess):
        return _indexed_elements(relation, call, ctx)
    if isinstance(relation, WindowAccess):
        return _windowed_elements(relation, call, ctx, boundary, output)
    return _affine_elements(relation, call, ctx)


def _shape_of(type_) -> tuple | None:
    """One boundary value's shape, or ``None`` when it does not have one."""
    return tuple(type_.shape) if isinstance(type_, TensorType) else None


def _indexed_elements(relation: IndexedAccess, call, ctx) -> AccessQuantity | None:
    """One source slice per index element, whatever the index elements hold."""
    if relation.index_operand >= len(call.args):
        return None
    indices = _elements(_shape_of(ctx.type_of(call.args[relation.index_operand])) or ())
    source = _shape_of(ctx.type_of(call.args[0]))
    if indices is None or source is None or not 0 <= relation.source_axis < len(source):
        return None
    rest = _elements(
        tuple(
            extent for axis, extent in enumerate(source) if axis != relation.source_axis
        )
    )
    if rest is None:
        return None
    return AccessQuantity(indices * rest, indices * rest)


def _windowed_elements(
    relation: WindowAccess, call, ctx, boundary: int, output: bool
) -> AccessQuantity | None:
    """The window itself, or the container without it."""
    holder = ctx.type_of(call) if output else ctx.type_of(call.args[boundary])
    container = _shape_of(holder)
    if container is None:
        return None
    whole = _elements(container)
    span = _window_span(relation, container, ctx, call)
    if whole is None or span is None:
        return None
    low, high = span
    if relation.complement:
        return AccessQuantity(max(whole - high, 0), max(whole - low, 0))
    return AccessQuantity(low, high)


def _affine_elements(relation, call, ctx) -> AccessQuantity | None:
    """How many accesses the relation itself states over the result domain.

    The domain is one point per element of the result, and the relation maps
    each of those to what it reads. Counting the relation rather than the
    operand is what makes a many-to-one read cost less than the operand and a
    broadcast cost more than one point.
    """
    domain = _shape_of(ctx.type_of(call))
    points = _elements(domain or ())
    if points is None:
        return None
    if isinstance(relation, isl.multi_aff):
        return AccessQuantity(points, points)
    if not isinstance(relation, isl.map):
        return None
    reached = relation.intersect_domain(_domain_set(domain or ()))
    counted = int(str(reached.wrap().count_val()))
    return AccessQuantity(counted, counted)


def _domain_set(shape: tuple) -> "isl.set":
    """One iteration point per element of *shape*."""
    if not shape:
        return isl.set("{ [] }")
    dims = ", ".join(f"d{index}" for index in range(len(shape)))
    bounds = " and ".join(f"0 <= d{index} < {extent}" for index, extent in enumerate(shape))
    return isl.set(f"{{ [{dims}] : {bounds} }}")


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


def forward_whole(call, operand: int, ctx) -> StorageEffectClaim:
    """Forward all of *operand*, with its address when the sizes are static."""
    size = static_bytes(ctx.type_of(call))
    if size is None or size != static_bytes(ctx.type_of(call.args[operand])):
        return StorageEffectClaim(StorageEffectKind.FORWARD, (operand,))
    return StorageEffectClaim(
        StorageEffectKind.FORWARD, (operand,), (StorageSpan(operand, 0, size),)
    )


def update_destination(call, ctx, *, destination: int) -> "StorageEffectClaim | None":
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
    return StorageEffectClaim(
        StorageEffectKind.UPDATE, (destination,), (StorageSpan(destination, 0, size),)
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
    "StorageEffectClaim",
    "StorageEffectKind",
    "StorageSpan",
    "declared_storage",
    "dense",
    "forward_whole",
    "same_placement",
    "static_bytes",
    "update_destination",
    "AccessQuantity",
    "IndexedAccess",
    "OperandValue",
    "access_elements",
    "WindowAccess",
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
