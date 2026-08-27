"""Per-analysis Context dataclasses.

TypeInferContext is the type-inference dispatch + unified error helper. VerifyContext
extends it with a mesh scope stack. CostContext seeds recursive-local Cost
Evaluators with the selected candidate's input/output Types.

The concrete CUDA CodegenContext lives in tilefoundry.codegen.cuda.context —
this module only needs the generic contract, so codegen-side context is
imported indirectly (no cycle).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import NoReturn, Union

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.expr import Call, Expr
from tilefoundry.ir.core.metadata import (
    BindingMetadata,
    SourceSpanMetadata,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.stmt import Stmt
from tilefoundry.ir.types.shard import Topology
from tilefoundry.ir.types.tensor_type import DType, Type
from tilefoundry.ir.types.utils import local_type_of


@dataclass(frozen=True)
class FunctionScope:
    """Where a walk is reading: one Module tree, and whose body it is in.

    A `Function` carries no execution context and one object is reachable from
    more than one program, so anything answered about the function being read --
    which Module owns it, what it may reach -- is answered within the tree the
    walk was given rather than globally.
    """

    module: Module
    function: Function


@dataclass
class TypeInferContext:
    """Track walk location and route type-inference queries.

    Derivation lives in ``TypeInferVisitor``. ``scope`` says where the walk is
    reading. ``mesh_scope`` carries enclosing scopes to statement verifiers
    without generic verification importing operation classes.
    See [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).
    """

    scope: FunctionScope | None = None
    mesh_scope: tuple = ()
    memo: dict[int, tuple[Expr, Type]] = field(default_factory=dict, repr=False, compare=False)
    instantiated_memo: dict[tuple[int, tuple[Type, ...]], Type] = field(
        default_factory=dict, repr=False, compare=False
    )

    def child_for(self, callee: object):
        """Return the direct child module that owns *callee*, if any."""
        if self.scope is None or self.scope.module is None:
            return None
        from tilefoundry.ir.core.module import child_module_of  # noqa: PLC0415

        return child_module_of(self.scope.module, self.scope.function, callee)

    def scope_for(self, callee: object) -> FunctionScope | None:
        """Return the runtime scope in which *callee*'s body is read."""
        if self.scope is None:
            child = self.child_for(callee)
            return None if child is None else FunctionScope(child, callee)
        child = self.child_for(callee)
        return FunctionScope(child or self.scope.module, callee)

    def for_callee(self, callee: object) -> TypeInferContext:
        """Move to *callee* with a fresh scope memo and the shared call cache."""
        return replace(self, scope=self.scope_for(callee), memo={})

    def type_of(self, expr: Expr) -> Type:
        """Read a bound type from this scope, falling back to the node type."""
        hit = self.memo.get(id(expr))
        return hit[1] if hit is not None else expr.type

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
            loc = node.loc
            where = f"\n  at {loc}" if loc else ""
        raise VerifyError(f"{name}: {msg}{where}")


@dataclass
class VerifyContext(TypeInferContext):
    """Extends TypeInferContext with a mesh scope stack.

    VerifyVisitor pushes/pops the enclosing `MeshScope.mesh` as it traverses,
    so per-stmt verify handlers can check that any `ShardLayout.mesh`
    referenced at the current point is in scope (see [tir §1.3](docs/spec/tir.md#13-primfunction)).
    """

    mesh_stack: list = field(default_factory=list)


@dataclass
class CostContext(TypeInferContext):
    """Cost Evaluator context for one topology window.

    ``level=None`` exposes the types as written. A named level projects them to
    what one unit of that level holds, letting the same registered evaluator
    answer both global and per-unit questions.
    """

    selected_types: Mapping[int, Type] = field(default_factory=dict)
    selected_output_type: Type | None = None
    level: str | None = None
    topologies: tuple[Topology, ...] = ()

    def type_of(self, expr: Expr) -> Type:
        selected = self.selected_types.get(id(expr))
        return selected if selected is not None else super().type_of(expr)

    def local_type_of(self, expr: Expr) -> Type:
        """Return ``expr``'s Type in this context's topology window."""
        selected = self.selected_types.get(id(expr))
        type_ = selected if selected is not None else expr.type
        if self.level is None:
            return type_
        return local_type_of(type_, level=self.level, topologies=self.topologies)

    def local_output_type(self, call: Call) -> Type:
        """Return the selected candidate output in recursive-local form."""
        output = self.selected_output_type
        if output is None:
            output = call.type
        if self.level is None:
            return output
        return local_type_of(output, level=self.level, topologies=self.topologies)


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
    "FunctionScope",
    "TypeInferContext",
    "VerifyContext",
    "CostContext",
    "Cost",
    "TrafficBytes",
]
