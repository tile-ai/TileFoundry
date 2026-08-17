"""Register per-operation proofs that a result reuses an operand's buffer.

An Op says which of its operands its result lives in, and where inside them when
it can address that far; this layer follows those claims along the operand edges
until they reach a value that owns its bytes. Every step is fail-closed -- no
handler, unreadable facts, operands that do not meet in one base, spans that do
not cover the result -- because over-reporting one allocation is safe and missing
one is not. A claim without spans still concludes, but states no address, so it
cannot be a piece of somebody else's coverage.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tilefoundry.ir.core.expr import Call, Constant, Expr, Op
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of

from .registries import AnalysisRegistry


class AliasKind(enum.Enum):
    """What one Op claims about where its result's bytes live."""

    PRODUCE = "produce"
    FORWARD = "forward"
    UPDATE = "update"


@dataclass(frozen=True)
class AliasSpan:
    """One run of bytes inside one operand's own buffer."""

    operand: int
    offset: int
    size: int


@dataclass(frozen=True)
class AliasClaim:
    """Which operands a result lives in, and where inside them if that is known."""

    operands: tuple[int, ...]
    spans: tuple[AliasSpan, ...] = ()


@dataclass(frozen=True)
class AliasExtent:
    """Where one value's bytes sit, as an offset and length in a base buffer."""

    base: Expr
    offset: int
    size: int


@dataclass(frozen=True)
class AliasHandler:
    """One Op's alias proof, and the conclusion it draws when the proof closes."""

    kind: AliasKind
    prove: Callable[[Call, "AliasContext"], AliasClaim | None]
    destination: int | None = None


@dataclass
class AliasContext:
    """The one Function walk a set of alias proofs is made against."""

    type_of: Callable[[Expr], Type]
    users: Mapping[int, tuple[Expr, ...]] = field(default_factory=dict)
    positions: Mapping[int, int] = field(default_factory=dict)
    caller_owned: frozenset[int] = frozenset()
    extents: dict[int, AliasExtent] = field(default_factory=dict)
    members: dict[int, list[Expr]] = field(default_factory=dict)

    def record(self, expr: Expr, base: Expr, extent: AliasExtent | None = None) -> None:
        """Remember that *expr*'s bytes belong to *base*, at *extent* if known."""
        if extent is not None:
            self.extents[id(expr)] = extent
        self.members.setdefault(id(base), []).append(expr)

    def bytes_of(self, expr: Expr) -> int | None:
        """How many bytes *expr* holds, or ``None`` when that is not static."""
        return static_bytes(self.type_of(expr))

    def extent_of(self, expr: Expr) -> AliasExtent | None:
        """Where *expr* lives: a proven span of a base, or its own buffer."""
        known = self.extents.get(id(expr))
        if known is not None:
            return known
        size = self.bytes_of(expr)
        return None if size is None else AliasExtent(expr, 0, size)

    def base_of(self, expr: Expr) -> Expr | None:
        """The value that owns the bytes *expr* reads."""
        extent = self.extent_of(expr)
        return None if extent is None else extent.base

    def last_authored_use(self, base: Expr, call: Call) -> bool:
        """Whether *call* is the last authored reader of *base*'s alias group."""
        here = self.positions.get(id(call))
        if here is None:
            return False
        for member in (base, *self.members.get(id(base), ())):
            for user in self.users.get(id(member), ()):
                if user is not call and self.positions.get(id(user), -1) > here:
                    return False
        return True


alias_registry: AnalysisRegistry = AnalysisRegistry("alias")


def register_alias(
    op_cls: type, kind: AliasKind, *, destination: int | None = None
) -> Callable[[Callable], Callable]:
    """Declare how one Op's result relates to its operands' buffers."""

    def decorate(prove: Callable) -> Callable:
        alias_registry.register(op_cls, AliasHandler(kind, prove, destination))
        return prove

    return decorate


def declared_alias(target: Op) -> AliasHandler | None:
    """What this Op claims before any proof is attempted."""
    return alias_registry.lookup(type(target))


def static_bytes(type_: Type) -> int | None:
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


def dense(type_: Type) -> bool:
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


def same_placement(left: Type, right: Type) -> bool:
    """Whether two Types name the same storage, element size, and positions."""
    if not (isinstance(left, TensorType) and isinstance(right, TensorType)):
        return False
    if left.storage != right.storage or left.dtype != right.dtype:
        return False
    left_shard, right_shard = shard_layout_of(left.layout), shard_layout_of(right.layout)
    if left_shard is None or right_shard is None:
        return left_shard is None and right_shard is None
    return left_shard.mesh == right_shard.mesh


def forward_whole(call: Call, operand: int, ctx: AliasContext) -> AliasClaim:
    """Forward all of *operand*, with its address when the sizes are static."""
    size = ctx.bytes_of(call)
    if size is None or size != ctx.bytes_of(call.args[operand]):
        return AliasClaim((operand,))
    return AliasClaim((operand,), (AliasSpan(operand, 0, size),))


def update_in_place(
    call: Call, ctx: AliasContext, *, destination: int, source: int
) -> AliasClaim | None:
    """Claim that the result is the destination's buffer after an overwrite.

    Overwriting is only free where nothing else still needs what was there: the
    destination must be storage this function may reuse, its whole alias group
    must be done reading by this call, and the bytes being written must come
    from somewhere else. A caller's parameter is not this function's to reuse,
    so it is refused here rather than donated.
    """
    if not same_placement(ctx.type_of(call.args[destination]), ctx.type_of(call)):
        return None
    size = ctx.bytes_of(call)
    if size is None or size != ctx.bytes_of(call.args[destination]):
        return None
    base = ctx.base_of(call.args[destination])
    if base is None or id(base) in ctx.caller_owned or isinstance(base, Constant):
        return None
    if ctx.base_of(call.args[source]) is base:
        return None
    if not ctx.last_authored_use(base, call):
        return None
    return AliasClaim((destination,), (AliasSpan(destination, 0, size),))


def prove_alias(call: Call, ctx: AliasContext) -> tuple[AliasKind, tuple[int, ...]] | None:
    """Ask the Op what it claims, and hold that claim to one base."""
    handler = declared_alias(call.target)
    if handler is None:
        return None
    try:
        claim = handler.prove(call, ctx)
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if claim is None or not claim.operands:
        return None
    if claim.spans:
        extent = compose(call, claim, ctx)
        if extent is None:
            return None
        ctx.record(call, extent.base, extent)
        return handler.kind, claim.operands
    base = _one_base(call, claim.operands, ctx)
    if base is None:
        return None
    ctx.record(call, base)
    return handler.kind, claim.operands


def _one_base(call: Call, operands: tuple[int, ...], ctx: AliasContext) -> Expr | None:
    """The single base every named operand resolves to, if there is one."""
    bases = []
    for operand in operands:
        if not 0 <= operand < len(call.args):
            return None
        base = ctx.base_of(call.args[operand])
        if base is None:
            return None
        bases.append(base)
    first = bases[0]
    return first if all(base is first for base in bases) else None


def compose(call: Call, claim: AliasClaim, ctx: AliasContext) -> AliasExtent | None:
    """Resolve a claim's spans onto one base, covering the result exactly."""
    result_bytes = ctx.bytes_of(call)
    if result_bytes is None:
        return None
    resolved: list[tuple[Expr, int, int]] = []
    for span in claim.spans:
        if not 0 <= span.operand < len(call.args) or span.offset < 0 or span.size <= 0:
            return None
        operand = ctx.extent_of(call.args[span.operand])
        if operand is None or span.offset + span.size > operand.size:
            return None
        resolved.append((operand.base, operand.offset + span.offset, span.size))
    base = resolved[0][0]
    if any(item[0] is not base for item in resolved):
        return None
    pieces = sorted((offset, size) for _, offset, size in resolved)
    cursor = pieces[0][0]
    for offset, size in pieces:
        if offset != cursor:
            return None
        cursor += size
    if cursor - pieces[0][0] != result_bytes:
        return None
    return AliasExtent(base, pieces[0][0], result_bytes)


__all__ = [
    "AliasClaim",
    "AliasContext",
    "AliasExtent",
    "AliasHandler",
    "AliasKind",
    "AliasSpan",
    "alias_registry",
    "declared_alias",
    "dense",
    "forward_whole",
    "prove_alias",
    "register_alias",
    "same_placement",
    "static_bytes",
    "update_in_place",
]
