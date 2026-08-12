"""Per-analysis Context dataclasses.

TypeInferContext is the type-of-cache + unified error helper. VerifyContext
extends it with a mesh scope stack. CostContext seeds recursive-local Cost
Evaluators with the selected candidate's input/output Types.

The concrete CUDA CodegenContext lives in tilefoundry.codegen.cuda.context —
this module only needs the generic contract, so codegen-side context is
imported indirectly (no cycle).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NoReturn, Union

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
from tilefoundry.ir.types.tensor_type import DType, TensorType, Type
from tilefoundry.ir.types.utils import local_type_of


def _constant_type(value: object) -> TensorType:
    if isinstance(value, bool):
        dtype = DType.bool
    elif isinstance(value, int):
        dtype = DType.i64
    elif isinstance(value, float):
        dtype = DType.f32
    else:
        raise VerifyError(f"Constant: unsupported value type {type(value).__name__}")
    return TensorType.umat_scalar(dtype)


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
    """Cache walk-local types and format type-inference errors.

    Derivation lives in ``TypeInferVisitor``. ``scope`` says where the walk is
    reading. ``mesh_scope`` carries enclosing scopes to statement verifiers
    without generic verification importing operation classes.
    ``elaboration_cache`` memoizes function instances by template identity and
    argument types for one parse or elaboration walk.
    See [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).
    """

    scope: FunctionScope | None = None
    cache: dict[int, Type] = field(default_factory=dict)
    mesh_scope: tuple = ()
    elaboration_cache: dict[tuple, Any] = field(default_factory=dict)

    def type_of(self, expr: Expr) -> Type:
        key = id(expr)
        cached = self.cache.get(key)
        if cached is not None:
            return cached


        from .visitors import TypeInferVisitor  # noqa: PLC0415

        computed = TypeInferVisitor(self).visit(expr)
        self.cache[key] = computed
        return computed

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
        if selected is not None:
            return selected
        return super().type_of(expr)

    def local_type_of(self, expr: Expr) -> Type:
        """Return ``expr``'s Type in this context's topology window."""
        type_ = self.type_of(expr)
        if self.level is None:
            return type_
        return local_type_of(type_, level=self.level, topologies=self.topologies)

    def local_output_type(self, call: Call) -> Type:
        """Return the selected candidate output in recursive-local form."""
        output = self.selected_output_type
        if output is None:
            output = self.type_of(call)
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

    ``flops`` groups leaf-local logical work by compute ``DType`` so one Op
    can report mixed work without selecting an ALU/TensorCore
    implementation. ``traffic`` carries one entry per operand in call order
    with the result last, so an Op that reads part of an input says so where
    it knows it. Neither field selects a hardware implementation, and neither
    names a memory level: that is a function of the operand's Type.
    """

    flops: Mapping[DType, int]
    traffic: tuple[TrafficBytes, ...]

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


__all__ = [
    "FunctionScope",
    "TypeInferContext",
    "VerifyContext",
    "CostContext",
    "Cost",
    "TrafficBytes",
]
