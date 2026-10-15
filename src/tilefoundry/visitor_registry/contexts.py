"""Per-analysis context dataclasses used by registry visitors."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import NoReturn, Union

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.expr import Call, Expr
from tilefoundry.ir.core.metadata import (
    BindingMetadata,
    SourceSpanMetadata,
    ValuationProvider,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.stmt import Stmt
from tilefoundry.ir.types import Mesh, Topology
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.types.tensor_type import DType, Type
from tilefoundry.ir.types.utils import local_type_of


@dataclass(frozen=True)
class TypeInferResults:
    """The type and optional value range produced by one inference rule."""

    type: Type
    value_range: tuple[int, int] | None = None


@dataclass
class TypeInferContext:
    """Track walk location and route type-inference queries.

    Derivation lives in ``TypeInferVisitor``. ``frame`` says where the walk is
    reading. ``current_mesh`` is the composed HIR execution domain.
    See [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).
    """

    frame: CallFrame | None = None
    current_mesh: Mesh | None = None
    memo: dict[int, tuple[Expr, Type]] = field(default_factory=dict, repr=False, compare=False)
    instantiated_memo: dict[tuple, Type] = field(
        default_factory=dict, repr=False, compare=False
    )
    loops: tuple[tuple[Expr, object, object, object], ...] = ()
    loop_aliases: dict[int, Expr] = field(default_factory=dict, repr=False, compare=False)

    call: Call | None = None
    owner: Expr | None = None
    parent: TypeInferContext | None = field(default=None, repr=False, compare=False)

    def with_captures(self, captures):
        """Keep captured induction parameters in their original loop coordinate."""
        aliases = dict(self.loop_aliases)
        for param, arg in captures:
            aliases[id(param)] = self.capture_root(arg)
        return replace(self, loop_aliases=aliases)

    def for_region(self, region):
        """Enter a region after its arguments have been visited outside it."""
        from tilefoundry.ir.hir.loop_region import LoopRegion  # noqa: PLC0415

        inner = self.with_captures(region.captures())
        if isinstance(region, LoopRegion):
            loop = (region.induction_var, region.start, region.extent, region.step)
            return replace(inner, loops=(*self.loops, loop), owner=region, parent=self)
        mesh = make_mesh(self.current_mesh, region.mesh) if self.current_mesh else region.mesh
        return replace(inner, current_mesh=mesh, owner=region, parent=self)

    def for_call(self, call):
        """Interpret a call in its caller's frame."""
        return replace(self, call=call)

    def capture_root(self, expr):
        """Resolve captures without treating carried values as invariant aliases."""
        return self.loop_aliases.get(id(expr), expr)

    def ancestors(self):
        """Lexical regions, from the innermost context to the definition root."""
        current = self
        while current is not None:
            yield current
            current = current.parent

    def enclosing_loops(self):
        """Loop owners in their lexical order."""
        from tilefoundry.ir.hir.loop_region import LoopRegion  # noqa: PLC0415

        return tuple(
            reversed([ctx.owner for ctx in self.ancestors() if isinstance(ctx.owner, LoopRegion)])
        )

    def enclosing_mesh(self):
        """The nearest authored mesh, or the definition's external mesh."""
        from tilefoundry.ir.hir.mesh_region import MeshRegion  # noqa: PLC0415

        return next(
            (ctx.owner.mesh for ctx in self.ancestors() if isinstance(ctx.owner, MeshRegion)),
            self.current_mesh,
        )

    def resolve_target(self):
        """The target carried by this visitor's current call frame."""
        return None if self.frame is None else self.frame.target

    def for_callee(self, callee: object, *, frame=None) -> TypeInferContext:
        """Enter a same-module function with fresh type bindings."""
        if frame is None and self.frame is not None:
            frame = replace(self.frame, function=callee)
        return replace(self, frame=frame, owner=callee, parent=None, call=None, memo={})

    def type_of(self, expr: Expr) -> Type:
        """The type this inference derived or bound for *expr*.

        Inference reads only its own memo: a type stored on the node by an
        earlier walk is not an input, so a value this walk never reached is an
        error rather than a stale answer.
        """
        hit = self.memo.get(id(expr))
        if hit is None:
            self.error(expr, "this inference has no type for the value; it was not visited")
        return hit[1]

    def local_type_of(self, expr: Expr) -> Type:
        """Read an expression type without topology projection."""
        return self.type_of(expr)

    def error(self, node: Union[Expr, Stmt], msg: str) -> NoReturn:
        if isinstance(node, Call):
            name = type(node.target).__name__
        else:
            name = type(node).__name__
        binding = get_metadata(node, BindingMetadata) if isinstance(node, Expr) else None
        span = get_metadata(node, SourceSpanMetadata) if isinstance(node, Expr) else None
        if span is not None:
            label = f" variable {binding.name!r}" if binding is not None else ""
            where = f"\n  at {span.file}:{span.line}:{span.column}{label}"
        elif isinstance(node, Expr):
            location = binding_name(node)
            where = f"\n  at {location}" if location else ""
        else:
            loc = getattr(node, "loc", None)
            where = f"\n  at {loc}" if loc else ""
        raise VerifyError(f"{name}: {msg}{where}")


class ValuationContext(TypeInferContext, ValuationProvider):
    """Provide the interpretations implemented by the current family context."""

    def request_valuation[V](self, valuation: type[V]) -> V:
        if isinstance(self, valuation):
            return self
        self.error(
            self.call or self.owner, f"{type(self).__name__} does not provide {valuation.__name__}"
        )


@dataclass
class VerifyContext(ValuationContext):
    """The context a per-stmt verify handler is annotated against.

    The enclosing scope a handler asks about is the TIR ``mesh_scope``
    tuple, set by whoever walks the statements
    (see [tir §1.3](docs/spec/tir.md#13-primfunction)).
    """

    mesh_scope: tuple = ()

    def type_of(self, expr: Expr) -> Type:
        """The type the checked IR already holds: a check reads, it does not infer."""
        return expr.type


@dataclass
class CostContext(ValuationContext):
    """Cost Evaluator context for one topology window.

    ``topology_level=None`` exposes the types as written. A named level projects
    them to what one unit of that level holds, letting the same registered
    evaluator answer both global and per-unit questions.
    """

    selected_types: Mapping[int, Type] = field(default_factory=dict)
    selected_output_type: Type | None = None
    topology_level: str | None = None
    topologies: tuple[Topology, ...] = ()

    def type_of(self, expr: Expr) -> Type:
        selected = self.selected_types.get(id(expr))
        return selected if selected is not None else expr.type

    def local_type_of(self, expr: Expr) -> Type:
        """Return ``expr``'s Type in this context's topology window."""
        selected = self.selected_types.get(id(expr))
        type_ = selected if selected is not None else expr.type
        if self.topology_level is None:
            return type_
        return local_type_of(type_, topology_level=self.topology_level, topologies=self.topologies)

    def local_output_type(self, call: Call) -> Type:
        """Return the selected candidate output in recursive-local form."""
        output = self.selected_output_type
        if output is None:
            output = call.type
        if self.topology_level is None:
            return output
        return local_type_of(output, topology_level=self.topology_level, topologies=self.topologies)


@dataclass(frozen=True)
class TrafficBytes:
    """Bytes one operand moves, read and write kept apart."""

    read: int = 0
    write: int = 0

    @property
    def total_bytes(self) -> int:
        """Bytes moved in either direction."""
        return self.read + self.write


@dataclass(frozen=True)
class Cost:
    """Leaf-local logical work for one selected ``OpCandidate``.

    ``flops`` groups leaf-local logical work by compute ``DType`` so one Op can
    report mixed work without selecting an ALU/TensorCore implementation.
    ``service`` counts what is not floating point at all -- a comparison, a
    select, an integer add -- by the service it asks for, because a dtype is not
    a kind of work. ``traffic`` carries one entry per operand in call order with
    the result last, so an Op that reads part of an input says so where it knows
    it. No field names a hardware implementation or a memory level.
    """

    flops: Mapping[DType, int]
    traffic: tuple[TrafficBytes, ...]
    service: Mapping[str, int] = field(default_factory=dict)
    sent: tuple[tuple[str, TrafficBytes], ...] = ()
    """Bytes that left the unit they were on, by the boundary they crossed.

    ``traffic`` says which storage level each operand touched. Data handed
    between two units of one topology level touches the same storage at both
    ends and has still gone somewhere, so it is stated against that level's
    name in addition to the storage traffic, from the sending unit's own view.
    """

    @property
    def ops(self) -> Mapping[str, int]:
        """Program-work spelling for non-floating-point operation counts."""
        return self.service

    @property
    def bytes(self) -> int:
        """Every operand's traffic, in either direction."""
        return sum(moved.total_bytes for moved in self.traffic)

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in self.flops.values()):
            raise ValueError("Cost flops must be non-negative integers")
        if any(
            not isinstance(value, int) or value < 0
            for moved in self.traffic
            for value in (moved.read, moved.write)
        ):
            raise ValueError("Cost traffic must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.service.values()
        ):
            raise ValueError("Cost service work must be non-negative integers")


__all__ = [
    "TypeInferContext",
    "ValuationContext",
    "VerifyContext",
    "CostContext",
    "Cost",
    "TrafficBytes",
]
