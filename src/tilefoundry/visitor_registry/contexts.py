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
from tilefoundry.ir.core.stmt import Stmt
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
    return TensorType.meta_scalar(dtype)


@dataclass
class TypeInferContext:
    """Walk-local type-of cache + error helper. Spec §4.

    The actual per-``Expr``-kind derivation rules live on
    ``TypeInferVisitor`` (visitor_registry.visitors); this context is only
    the memo dict and the shared ``error()`` formatter — it does not
    dispatch on ``type(expr)`` itself.

    ``mesh_scope`` carries the enclosing ``MeshScope`` stack into a registered
    ``verify_stmt`` handler (the stmt walk sets it before dispatch), so a
    mesh-scoped op (``Mma`` atom-scope, ``Sync``) can verify against its
    enclosing meshes without the generic verify importing those op classes.

    ``elaboration_cache`` is ``ir.hir.function.elaborate``'s
    (template id, arg types) -> instance memo, shared by reference across
    one elaboration walk (and by the parser across one parse session) so
    repeated call sites collapse onto the same instance.
    """

    module: Any = None
    cache: dict[Expr, Type] = field(default_factory=dict)
    mesh_scope: tuple = ()
    elaboration_cache: dict[tuple, Any] = field(default_factory=dict)

    def type_of(self, expr: Expr) -> Type:
        cached = self.cache.get(expr)
        if cached is not None:
            return cached
        # Local import: visitors.py imports TypeInferContext from this module,
        # so the reverse import is deferred to call time to avoid a cycle.
        from .visitors import TypeInferVisitor  # noqa: PLC0415

        computed = TypeInferVisitor(self).visit(expr)
        self.cache[expr] = computed
        return computed

    def error(self, node: Union[Expr, Stmt], msg: str) -> NoReturn:
        if isinstance(node, Call):
            name = type(node.target).__name__
        else:
            name = type(node).__name__
        loc = getattr(node, "loc", None)
        where = f"\n  at {loc}" if loc else ""
        raise VerifyError(f"{name}: {msg}{where}")


@dataclass
class VerifyContext(TypeInferContext):
    """Extends TypeInferContext with a mesh scope stack.

    VerifyVisitor pushes/pops the enclosing `MeshScope.mesh` as it traverses,
    so per-stmt verify handlers can check that any `ShardLayout.mesh`
    referenced at the current point is in scope (see tir spec §6.6).
    """

    mesh_stack: list = field(default_factory=list)


@dataclass
class CostContext(TypeInferContext):
    """Recursive-local Cost Evaluator context.

    A Cost Evaluator needs no active-topology selector: ``local_type_of``
    projects every already-resolved nested ``ShardLayout`` exactly once, so
    the same registered handler returns per-GPU work for a GPU-local Type
    and per-CTA work for the corresponding nested GPU-plus-CTA Type.
    """

    selected_types: Mapping[int, Type] = field(default_factory=dict)
    selected_output_type: Type | None = None

    def type_of(self, expr: Expr) -> Type:
        selected = self.selected_types.get(id(expr))
        if selected is not None:
            return selected
        return super().type_of(expr)

    def local_type_of(self, expr: Expr) -> Type:
        """Return ``expr``'s recursive-local Type (thin wrapper over the
        shared ``ir.types.utils.local_type_of`` projection)."""
        return local_type_of(self.type_of(expr))

    def local_output_type(self, call: Call) -> Type:
        """Return the selected candidate output in recursive-local form."""
        output = self.selected_output_type
        if output is None:
            output = self.type_of(call)
        return local_type_of(output)


@dataclass(frozen=True)
class Cost:
    """Leaf-local logical work for one selected ``OpCandidate``.

    ``flops`` groups leaf-local logical work by compute ``DType`` so one Op
    can report mixed work without selecting an ALU/TensorCore
    implementation. ``bytes`` is scalar logical byte traffic. Neither field
    selects a hardware implementation.
    """

    flops: Mapping[DType, int]
    bytes: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in self.flops.values()):
            raise ValueError("Cost flops must be non-negative integers")
        if not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError("Cost bytes must be a non-negative integer")


__all__ = [
    "TypeInferContext",
    "VerifyContext",
    "CostContext",
    "Cost",
]
