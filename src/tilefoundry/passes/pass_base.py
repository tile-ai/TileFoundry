"""Pass / FunctionPass / PrimFuncPass / ModulePass base classes.

(frozen Module in, new Module out).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction


class Pass(ABC):
    """Base class for every pass.

    Subclasses set class attributes:
    - ``name``     — short identifier used for dump/log.
    - ``requires`` — tuple of pass names that must precede this one (MVP
                     does order-assert, not topological sort).
    """

    name: str = ""
    requires: tuple[str, ...] = ()

    @abstractmethod
    def run(self, module: Module) -> Module:
        ...


class ModulePass(Pass):
    """Whole-module pass. Free to add/remove/reorder Functions."""


class FunctionPass(Pass):
    """Per-hir-Function pass.

    Per-hir-Function pass. Framework iterates `module.functions` and
    calls `run_function` on each `HirFunction`; tir entries pass through.
    """

    @abstractmethod
    def run_function(self, fn: HirFunction, module: Module) -> HirFunction:
        ...

    def run(self, module: Module) -> Module:
        new_fns = []
        changed = False
        for fn in module.functions:
            if isinstance(fn, HirFunction):
                nf = self.run_function(fn, module)
                new_fns.append(nf)
                if nf is not fn:
                    changed = True
            else:
                new_fns.append(fn)
        if not changed:
            return module
        return replace(module, functions=tuple(new_fns))


class PrimFuncPass(Pass):
    """Per-tir-PrimFunction pass.

    Per-tir-PrimFunction pass. Framework iterates `module.functions` and
    calls `run_prim_func` on each `PrimFunction`; hir entries pass through.
    """

    @abstractmethod
    def run_prim_func(self, fn: PrimFunction, module: Module) -> PrimFunction:
        ...

    def run(self, module: Module) -> Module:
        new_fns = []
        changed = False
        for fn in module.functions:
            if isinstance(fn, PrimFunction):
                nf = self.run_prim_func(fn, module)
                new_fns.append(nf)
                if nf is not fn:
                    changed = True
            else:
                new_fns.append(fn)
        if not changed:
            return module
        return replace(module, functions=tuple(new_fns))


__all__ = ["Pass", "ModulePass", "FunctionPass", "PrimFuncPass"]
