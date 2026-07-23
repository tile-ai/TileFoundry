"""Canonical home for AnalysisRegistry + per-analysis registry instances."""

from __future__ import annotations

from typing import Callable


class AnalysisRegistry[Key]:
    """Class → handler map. Double registration raises; lookup miss returns None."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._map: dict[Key, Callable] = {}

    def register(self, cls: Key, fn: Callable) -> None:
        if cls in self._map:
            raise RuntimeError(f"{self.name}: {cls.__name__} already registered")
        self._map[cls] = fn

    def lookup(self, cls: Key) -> Callable | None:
        return self._map.get(cls)

    def has(self, cls: Key) -> bool:
        return cls in self._map

    def decorator(self) -> Callable[[type], Callable[[Callable], Callable]]:
        """``@registry.decorator()`` factory: ``register_X = registry.decorator()``
        gives the conventional ``register_X(cls)`` decorator for this registry."""

        def register_for(cls: type) -> Callable[[Callable], Callable]:
            def decorator(fn: Callable) -> Callable:
                self.register(cls, fn)
                return fn

            return decorator

        return register_for


# Canonical per-analysis registries. Every instance is module-level so the
# @register_* decorators attach at import time.
typeinfer_registry: AnalysisRegistry = AnalysisRegistry("typeinfer")
verify_stmt_registry: AnalysisRegistry = AnalysisRegistry("verify_stmt")
codegen_cuda_registry: AnalysisRegistry = AnalysisRegistry("codegen_cuda")
codegen_cpu_registry: AnalysisRegistry = AnalysisRegistry("codegen_cpu")
cost_evaluator_registry: AnalysisRegistry = AnalysisRegistry("cost_evaluator")
# HIR op class → its HIR→TIR lowering handler. The HirToTir pass dispatches on
# ``type(call.target)`` through this registry instead of an isinstance chain, so
# a target-owned op (e.g. the CUDA MMA op) registers its own lowering.
hir_lowering_registry: AnalysisRegistry = AnalysisRegistry("hir_lowering")


register_typeinfer = typeinfer_registry.decorator()
register_verify_stmt = verify_stmt_registry.decorator()
register_codegen_cuda = codegen_cuda_registry.decorator()
register_codegen_cpu = codegen_cpu_registry.decorator()
register_cost_evaluator = cost_evaluator_registry.decorator()
register_hir_lowering = hir_lowering_registry.decorator()


__all__ = [
    "AnalysisRegistry",
    "typeinfer_registry",
    "verify_stmt_registry",
    "codegen_cuda_registry",
    "codegen_cpu_registry",
    "cost_evaluator_registry",
    "hir_lowering_registry",
    "register_typeinfer",
    "register_verify_stmt",
    "register_codegen_cuda",
    "register_codegen_cpu",
    "register_cost_evaluator",
    "register_hir_lowering",
]
