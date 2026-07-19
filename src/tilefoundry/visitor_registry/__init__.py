"""Visitor Registry — derived-visitor dispatch pattern.


To keep this package importable early (ir.core imports from here during
its own __init__), the package __init__ re-exports **only** the
lightweight registry bits. Contexts and Visitors live in submodules
``contexts`` and ``visitors`` and should be imported from there.
"""
from __future__ import annotations

from .registries import (
    AnalysisRegistry,
    codegen_cpu_registry,
    codegen_cuda_registry,
    cost_evaluator_registry,
    register_codegen_cpu,
    register_codegen_cuda,
    register_cost_evaluator,
    register_typeinfer,
    register_verify_stmt,
    typeinfer_registry,
    verify_stmt_registry,
)

__all__ = [
    "AnalysisRegistry",
    "typeinfer_registry",
    "verify_stmt_registry",
    "codegen_cuda_registry",
    "codegen_cpu_registry",
    "cost_evaluator_registry",
    "register_typeinfer",
    "register_verify_stmt",
    "register_codegen_cuda",
    "register_codegen_cpu",
    "register_cost_evaluator",
]
