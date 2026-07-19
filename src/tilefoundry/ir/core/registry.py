"""Back-compat re-exports. Canonical home is tilefoundry.visitor_registry.

Kept here so existing `from tilefoundry.ir.core.registry import register_typeinfer`
call-sites keep working. New code should import from
`tilefoundry.visitor_registry` directly.
"""
from __future__ import annotations

from tilefoundry.visitor_registry.registries import (
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

# Legacy names — pre-spec-rename, when codegen registries were called
# `lower_<target>_registry`. Kept as aliases; no new code should use them.
lower_cuda_registry = codegen_cuda_registry
lower_cpu_registry = codegen_cpu_registry
register_lower_cuda = register_codegen_cuda
register_lower_cpu = register_codegen_cpu

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
    # legacy
    "lower_cuda_registry",
    "lower_cpu_registry",
    "register_lower_cuda",
    "register_lower_cpu",
]
