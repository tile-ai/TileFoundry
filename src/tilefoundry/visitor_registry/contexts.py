"""Per-analysis Context dataclasses.

TypeInferContext is the type-of-cache + unified error helper. VerifyContext
extends it with a mesh scope stack. CostContext is a placeholder.

The concrete CUDA CodegenContext lives in tilefoundry.codegen.cuda.context —
this module only needs the generic contract, so codegen-side context is
imported indirectly (no cycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn, Union

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.expr import Call, Expr
from tilefoundry.ir.core.stmt import Stmt
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.ir.types.tensor_type import DType, TensorType, Type


def _constant_type(value: object) -> TensorType:
    if isinstance(value, bool):
        dtype = DType.bool
    elif isinstance(value, int):
        dtype = DType.i64
    elif isinstance(value, float):
        dtype = DType.f32
    else:
        raise VerifyError(f"Constant: unsupported value type {type(value).__name__}")
    return TensorType(shape=(), dtype=dtype, layout=EMPTY_LAYOUT, storage=None)


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
    """Cost-model context. MVP placeholder — no handlers registered."""


@dataclass
class Cost:
    """Placeholder cost record. Populated by future costmodel handlers."""

    flops: int = 0
    bytes: int = 0


__all__ = [
    "TypeInferContext",
    "VerifyContext",
    "CostContext",
    "Cost",
]
