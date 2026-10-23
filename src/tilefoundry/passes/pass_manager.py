"""Run a linear pass pipeline with automatic dirty-function rechecks.

Identity changes trigger HIR type inference or TIR verification with whole-
function fallback. Active dump scopes receive before and after IR snapshots;
disabled or absent scopes perform no I/O. See
[passes §5](docs/spec/passes.md#5-passmanager).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target.base import Target
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import (
    verify_function as verify_hir_function,
)
from tilefoundry.visitor_registry.verify import (
    verify_prim_function,
)

from .pass_base import ModulePass, Pass


class PassPoint(Enum):
    BEFORE_LOWER = "before_lower"
    AFTER_LOWER = "after_lower"
    BEFORE_CODEGEN = "before_codegen"


_TARGET_PASSES: dict[tuple[type[Target], PassPoint], list[type[ModulePass]]] = {}


def register_pass(target: type[Target], point: PassPoint) -> Callable[[type[ModulePass]], type[ModulePass]]:
    """Register a target's module pass at a fixed pipeline point."""
    def register(pass_type: type[ModulePass]) -> type[ModulePass]:
        passes = _TARGET_PASSES.setdefault((target, point), [])
        if any(pass_type in registered for (_, registered_point), registered in _TARGET_PASSES.items()
               if registered_point == point):
            raise ValueError(f"{target.name}: pass {pass_type.__name__} already registered at {point.value}")
        passes.append(pass_type)
        return pass_type
    return register


def target_passes(module: Module, point: PassPoint) -> list[ModulePass]:
    """Instantiate the module targets' passes, ordered by target name."""
    targets = {
        type(frame.function.target if isinstance(frame.function, PrimFunction) else frame.target)
        for frame in list_call_paths(module).values()
    }
    return [pass_type() for target in sorted(targets, key=lambda target: target.name)
            for pass_type in _TARGET_PASSES.get((target, point), ())]


@dataclass
class PassManager:
    passes: list[Pass] = field(default_factory=list)

    def add(self, p: Pass) -> "PassManager":
        self.passes.append(p)
        return self

    def run(self, module: Module) -> Module:
        self._check_requires()
        for seq, p in enumerate(self.passes):
            prev = module
            with DumpScope(f"{seq:02d}_{p.name}"):
                dump("before.txt", lambda: repr(prev), DumpFlags.PASS_IR)
                module = p.run(module)
                self._post_pass_recheck(prev, module)
                dump("after.txt", lambda: repr(module), DumpFlags.PASS_IR)
        return module

    def _post_pass_recheck(self, prev: Module, curr: Module) -> None:
        """Post pass recheck.

        Re-run typeinfer (HIR) / verify (TIR) on every function whose
        object identity changed between ``prev`` and ``curr``. Whole-
        function fallback per [passes §7](docs/spec/passes.md#7-implemented-passes).
        """
        previous = {id(frame.function) for frame in list_call_paths(prev).values()}
        for frame in list_call_paths(curr).values():
            fn = frame.function
            if id(fn) in previous:
                continue
            if isinstance(fn, HirFunction):
                verify_hir_function(fn)
                inference_type(
                    fn,
                    TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies),
                    ranges=True,
                )
            elif isinstance(fn, PrimFunction):
                verify_prim_function(fn, module_fns=curr)

    def _check_requires(self) -> None:
        seen: set[str] = set()
        for p in self.passes:
            for r in p.requires:
                if r not in seen:
                    raise ValueError(f"pass {p.name!r} requires {r!r} not registered before it")
            seen.add(p.name)


__all__ = ["PassManager", "PassPoint", "register_pass", "target_passes"]
