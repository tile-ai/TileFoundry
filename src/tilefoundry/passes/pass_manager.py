"""Run a linear pass pipeline with automatic dirty-function rechecks.

Identity changes trigger HIR type inference or TIR verification with whole-
function fallback. Active dump scopes receive before and after IR snapshots;
disabled or absent scopes perform no I/O. See
[passes §5](docs/spec/passes.md#5-passmanager).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.dump import DumpFlags, DumpScope, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.verify import verify_function as verify_hir_function
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.verify import verify_prim_function

from .pass_base import Pass


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
                dump("before.txt", repr(prev), DumpFlags.PASS_IR)
                module = p.run(module)
                self._post_pass_recheck(prev, module)
                dump("after.txt", repr(module), DumpFlags.PASS_IR)
        return module

    def _post_pass_recheck(self, prev: Module, curr: Module) -> None:
        """Post pass recheck.

        Re-run typeinfer (HIR) / verify (TIR) on every function whose
        object identity changed between ``prev`` and ``curr``. Whole-
        function fallback per [passes §7](docs/spec/passes.md#7-implemented-passes).
        """
        prev_by_name = {f.name: f for f in prev.functions}
        prim_fns = [f for f in curr.functions if isinstance(f, PrimFunction)]
        for fn in curr.functions:
            if prev_by_name.get(fn.name) is fn:
                continue
            if isinstance(fn, HirFunction):
                verify_hir_function(fn)
            elif isinstance(fn, PrimFunction):
                verify_prim_function(fn, module_fns=prim_fns)

    def _check_requires(self) -> None:
        seen: set[str] = set()
        for p in self.passes:
            for r in p.requires:
                if r not in seen:
                    raise ValueError(
                        f"pass {p.name!r} requires {r!r} not registered before it"
                    )
            seen.add(p.name)


__all__ = ["PassManager"]
