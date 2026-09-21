"""Visitor Registry — derived-visitor dispatch pattern.

To keep this package importable early (ir.core imports from here during
its own __init__), the package __init__ re-exports **only** the
lightweight registry bits. Contexts live in ``contexts``, type inference in
``typeinfer``, and the remaining derived visitors in ``visitors``; import each
from its owning submodule.
"""

from __future__ import annotations

from .registries import (
    DispatchRegistry,
    Role,
    codegen_registry,
    cost_evaluator_registry,
    register_codegen,
    register_cost_evaluator,
    register_typeinfer,
    register_verify_stmt,
    typeinfer_registry,
    verify_stmt_registry,
)

__all__ = [
    "DispatchRegistry",
    "Role",
    "codegen_registry",
    "cost_evaluator_registry",
    "register_codegen",
    "register_cost_evaluator",
    "register_typeinfer",
    "register_verify_stmt",
    "typeinfer_registry",
    "verify_stmt_registry",
]
