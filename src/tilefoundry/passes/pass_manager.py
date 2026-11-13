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

from tilefoundry import dispatch, dump, target, verifier
from tilefoundry.interfaces import typeinfer, valuation
from tilefoundry.ir import core

from .pass_base import ModulePass, Pass


class PassPoint(Enum):
    BEFORE_LOWER = "before_lower"
    AFTER_LOWER = "after_lower"
    BEFORE_CODEGEN = "before_codegen"


registry = dispatch.Registry("passes")
"""Target module passes keyed by ``(target class, pipeline point, pass class)``."""


def register_pass(tgt: type[target.Target], point: PassPoint) -> Callable[[type[ModulePass]], type[ModulePass]]:
    """Register a target's module pass at a fixed pipeline point."""
    def register(pass_type: type[ModulePass]) -> type[ModulePass]:
        if any(key[1] is point and key[2] is pass_type for key in registry.entries):
            raise ValueError(f"{tgt.name}: pass {pass_type.__name__} already registered at {point.value}")
        registry.register((tgt, point, pass_type), pass_type)
        return pass_type
    return register


def for_target(module: core.Module, point: PassPoint) -> list[ModulePass]:
    """Instantiate the module targets' passes, ordered by target name."""
    targets = set()
    for frame in valuation.collect_functions(module).values():
        tgt = frame.function.target if isinstance(frame.function, core.PrimFunction) else frame.target
        if tgt is not None:
            targets.add(type(tgt))
    return [pass_type() for tgt2 in sorted(targets, key=lambda tgt3: tgt3.name)
            for (key_target, key_point, _), pass_type in registry.entries.items()
            if key_target is tgt2 and key_point is point]


@dataclass
class PassManager:
    passes: list[Pass] = field(default_factory=list)

    def add(self, p: Pass) -> "PassManager":
        self.passes.append(p)
        return self

    def run(self, module: core.Module) -> core.Module:
        self.check_requires()
        for seq, p in enumerate(self.passes):
            prev = module
            previous = {id(frame.function) for frame in valuation.collect_functions(prev).values()}
            with dump.Scope(f"{seq:02d}_{p.name}"):
                dump.dump("before.txt", lambda: repr(prev), dump.Flags.PASS_IR)
                module = p.run(module)
                self.recheck(previous, module)
                dump.dump("after.txt", lambda: repr(module), dump.Flags.PASS_IR)
        return module

    def recheck(self, previous: set[int], curr: core.Module) -> None:
        """Post pass recheck.

        Re-run typeinfer (HIR) / verify (TIR) on every function whose
        object identity is absent from the pre-pass ``previous`` set, and on
        every function that reaches one of them by a call or a launch. Whole-
        function fallback per [passes §7](docs/spec/passes.md#7-implemented-passes).
        """
        frames = tuple(valuation.collect_functions(curr).values())
        changed = {id(frame.function) for frame in frames if id(frame.function) not in previous}
        for frame in frames:
            fn = frame.function
            if id(fn) in previous and not any(
                id(inner.function) in changed for inner in verifier.reached_frames(frame)
            ):
                continue
            if isinstance(fn, core.Function):
                verifier.verify_function(fn)
                typeinfer.inference_type(
                    fn,
                    typeinfer.TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies),
                    ranges=True,
                )
            elif isinstance(fn, core.PrimFunction):
                verifier.verify_prim_function(fn, frame=frame)

    def check_requires(self) -> None:
        seen: set[str] = set()
        for p in self.passes:
            for r in p.requires:
                if r not in seen:
                    raise ValueError(f"pass {p.name!r} requires {r!r} not registered before it")
            seen.add(p.name)


__all__ = ["PassManager", "PassPoint", "for_target", "register_pass", "registry"]
