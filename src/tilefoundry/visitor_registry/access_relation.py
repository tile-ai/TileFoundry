"""Register what one operation reads, writes, and where its result's bytes live.

Boundary handlers return one ``BoundaryAccess`` per boundary: a pattern saying
where it reads -- isl where that is affine, ``IndexedAccess`` or
``WindowAccess`` where a runtime value decides it -- and, separately, how much
it moves. Neither is inferred from the other.

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
    of ``index_operand``, so the address is not known here. Which operand is
    being indexed is said outright rather than assumed, because an Op may gather
    from more than one -- a rotation reads two tables through one set of
    positions.
    """

    source_operand: int
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


AccessPattern = Union["isl.multi_aff", "isl.map", IndexedAccess, WindowAccess]







@dataclass(frozen=True)
class AccessQuantity:
    """How many elements one boundary moves in one execution of an Op.

    ``lower`` and ``upper`` are equal whenever the Op can say the number. When
    they differ, ``provenance`` names the invariant the range came from, so a
    reader can check it rather than trust it; a prediction takes ``upper`` and
    says that it did.
    """

    lower: int
    upper: int
    provenance: str | None = None

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            edge = getattr(self, name)
            if not isinstance(edge, int) or isinstance(edge, bool) or edge < 0:
                raise ValueError(
                    f"an access moves a non-negative whole number of elements, "
                    f"and its {name} is {edge!r}"
                )
        if self.lower > self.upper:
            raise ValueError(
                f"an access of {self.lower}..{self.upper} elements runs backwards"
            )
        if self.lower != self.upper and not self.provenance:
            raise ValueError(
                f"an access of {self.lower}..{self.upper} elements is a range, so it "
                "must name the invariant it came from"
            )

    @property
    def exact(self) -> bool:
        """Whether one number answers this, rather than a range."""
        return self.lower == self.upper


@dataclass(frozen=True)
class BoundaryAccess:
    """One boundary: where it reads, and how much it moves.

    The two are separate answers. A pattern says which coordinates a value came
    from and is what a dependence needs; a quantity says how much crossed the
    boundary and is what a cost needs. Neither follows from the other -- every
    output of a scan depends on the whole input the scan reads once -- so the Op
    states both rather than having one inferred from the other.
    """

    pattern: AccessPattern
    quantity: AccessQuantity

    def __post_init__(self) -> None:
        if not isinstance(
            self.pattern, (isl.multi_aff, isl.map, IndexedAccess, WindowAccess)
        ):
            raise ValueError(
                f"a boundary reads through a relation, a lookup or a window, "
                f"not through {self.pattern!r}"
            )
        if not isinstance(self.quantity, AccessQuantity):
            raise ValueError(
                f"a boundary states how much it moves, not {self.quantity!r}"
            )


def moves(pattern: "AccessPattern", count: int) -> BoundaryAccess:
    """One boundary that moves a number of elements the Op can state."""
    return BoundaryAccess(pattern, AccessQuantity(count, count))


def moves_between(
    pattern: "AccessPattern", lower: int, upper: int, provenance: str
) -> BoundaryAccess:
    """One boundary whose amount an unbound value leaves within a range."""
    return BoundaryAccess(pattern, AccessQuantity(lower, upper, provenance))


def elements_of(type_: "Type") -> int:
    """How many elements one boundary value holds."""
    if not isinstance(type_, TensorType):
        raise ValueError(f"{type_!r} is not one tensor boundary")
    counted = 1
    for extent in type_.shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            raise ValueError(f"{type_!r} has no concrete element count")
        counted *= extent
    return counted


def access_elements(
    relations: AccessRelations, *, boundary: int, output: bool = False
) -> AccessQuantity | None:
    """What the Op said this boundary moves.

    Read back, never re-derived: the handler is the only thing that knows what
    its Op does, and a second opinion computed from the pattern would be a
    guess wearing the same type.
    """
    chosen = relations.outputs if output else relations.inputs
    return chosen[boundary].quantity if 0 <= boundary < len(chosen) else None


@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    One relation per boundary value, in boundary order.

    - ``inputs``: one `BoundaryAccess` per input arg, in argument order.
    - ``outputs``: one per output. Single-output ops have len 1; tuple-output
      ops have one entry per tuple field.
    - ``storage_effect``: where the result's bytes live. An Op that says nothing
      produces its own, because reading an operand at the same indices is not
      the same as being that operand.
    """

    inputs: tuple[BoundaryAccess, ...]
    outputs: tuple[BoundaryAccess, ...]
    storage_effect: "StorageEffectClaim | None" = None

    def __post_init__(self) -> None:
        for side in ("inputs", "outputs"):
            stated = getattr(self, side)
            if not isinstance(stated, tuple) or not all(
                isinstance(item, BoundaryAccess) for item in stated
            ):
                raise ValueError(f"{side} is one BoundaryAccess per boundary, got {stated!r}")
        if not self.outputs:
            raise ValueError("an operation produces at least one value to describe")







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


def _read_once(arg, ctx) -> int:
    """One whole operand, which is what an elementwise access moves."""
    type_ = ctx.type_of(arg)
    return elements_of(type_) if isinstance(type_, TensorType) else 0


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

        result = ctx.type_of(call)
        return AccessRelations(
            inputs=tuple(
                moves(_identity(_rank_of(call.args[index])), _read_once(call.args[index], ctx))
                for index in range(n_inputs)
            ),
            outputs=(moves(_identity(out_rank), elements_of(result)),),
            storage_effect=None if storage is None else storage(call, ctx),
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
    "AccessPattern",
    "AccessQuantity",
    "BoundaryAccess",
    "IndexedAccess",
    "elements_of",
    "moves",
    "moves_between",
    "OperandValue",
    "access_elements",
    "WindowAccess",
    "AccessRelations",
    "AccessRelationResult",
    "access_relation_registry",
    "type_relation_registry",
    "register_access_relation",
    "register_type_relation",
    "identity_relations",
    "build_relation",
]
