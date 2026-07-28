"""The shared vocabulary every model-driven test is written against.

One model is described once, as a `ModelCase`, and every kind of test reads that
one description: the reference run, the analyses, the schedules, the CLI, and the
end-to-end witnesses. Nothing here copies a model into a smaller graph for one
subsystem's convenience, because a result measured on a copy says nothing about
the program a user would actually hand us.

A case is target-free on purpose. The model source states shapes and dtypes; a
`TargetFixture` states the machine. They meet only when a test asks for it, so the
same model can be asked about on more than one target in one run.

`build()` re-executes the model source every time. That is not caution about
mutation in the abstract -- an analysis attaches its records to the Call objects
it measured, in place, so two tests sharing one built Module do not share a model,
they share each other's results. Binding a Target with `dataclasses.replace`
copies the Module value but keeps those very Call objects, so it isolates nothing
on its own; the fresh build has to come first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from tests.models.loader import load_model
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Target

MODELS_ROOT = Path(__file__).parent

Outcome = Literal["PASS", "BLOCKED"]


class CorpusError(ValueError):
    """A case describes something the corpus cannot resolve."""


@dataclass(frozen=True)
class CapabilityGate:
    """Whether one case is expected to work, and why not when it is not.

    A gate is only meaningful for a case somebody chose to run. A function nobody
    selected is untested, which is a different statement from blocked and must
    not be reported as one.
    """

    outcome: Outcome = "PASS"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome == "BLOCKED" and not self.reason.strip():
            raise CorpusError(
                "a blocked capability must state why; an unreasoned block cannot "
                "be reviewed or retired"
            )
        if self.outcome == "PASS" and self.reason:
            raise CorpusError(
                f"a passing capability states no reason, got {self.reason!r}"
            )

    @property
    def blocked(self) -> bool:
        return self.outcome == "BLOCKED"


@dataclass(frozen=True)
class TargetFixture:
    """One machine a model can be asked about, and the levels it declares.

    The topologies belong here rather than in the model source: how many parallel
    positions a program divides over is a property of the machine it was aimed
    at, and the same model is aimed at more than one.
    """

    id: str
    target: Target
    topologies: tuple[Topology, ...]

    def bind(self, module: Module) -> Module:
        """Return *module* aimed at this machine.

        The caller must pass a freshly built Module. This copies the Module value
        and keeps its Functions, which is correct for binding and useless for
        isolation.
        """
        return replace(module, target=self.target, topologies=self.topologies)

    def level(self, name: str) -> Topology:
        for topology in self.topologies:
            if topology.name == name:
                return topology
        declared = ", ".join(item.name for item in self.topologies) or "none"
        raise CorpusError(
            f"target fixture {self.id!r} declares no {name!r} level; it declares "
            f"{declared}"
        )


@dataclass(frozen=True)
class ReferenceCase:
    """The executable semantics one model is held to.

    `boundary` names what is actually run -- the whole decoder for a model small
    enough to run, or a complete submodule for one that is not. It is written
    down because a reference that quietly shrinks to a handful of leaf ops still
    reports PASS while proving much less.
    """

    id: str
    boundary: str
    entry: str
    inputs: Callable[..., object]
    oracle: Callable[..., object]
    problem_sizes: tuple[str, ...] = ()
    gate: CapabilityGate = field(default_factory=CapabilityGate)


@dataclass(frozen=True)
class FunctionCase:
    """One function of one model, selected to be analysed or scheduled."""

    id: str
    function: str
    problem_sizes: tuple[str, ...] = ()
    gate: CapabilityGate = field(default_factory=CapabilityGate)
    topology: str | None = None


@dataclass(frozen=True)
class ModelCase:
    """One model, described once, for every kind of test.

    `source` is re-executed on every `build()`; `namespace` is what that source
    is parameterised by. `entry` names the attribute the source leaves the Module
    in.
    """

    id: str
    source: Path
    entry: str
    namespace: Mapping[str, object] = field(default_factory=dict)
    reference: ReferenceCase | None = None
    analyze: tuple[FunctionCase, ...] = ()
    schedule: tuple[FunctionCase, ...] = ()

    def build(self) -> Module:
        """A Module nothing else holds a reference to.

        Re-executing the source is what makes this true. Two builds share no
        Function and no Call, so an analysis that annotates one is invisible to
        the other.
        """
        loaded = load_model(self.source, **dict(self.namespace))
        module = getattr(loaded, self.entry, None)
        if module is None:
            raise CorpusError(
                f"model case {self.id!r}: {self.source.name} defines no "
                f"{self.entry!r}"
            )
        if not isinstance(module, Module):
            raise CorpusError(
                f"model case {self.id!r}: {self.entry!r} is a "
                f"{type(module).__name__}, not a Module"
            )
        return module

    def build_for(self, fixture: TargetFixture) -> Module:
        """A fresh Module aimed at one machine, in the one order that isolates."""
        return fixture.bind(self.build())

    def inventory(self, module: Module | None = None) -> tuple[str, ...]:
        """Every HIR function this model really defines, in source order.

        Derived from a built Module rather than written down, so a function added
        to the model appears as untested instead of silently escaping the report.
        """
        built = self.build() if module is None else module
        return tuple(
            function.name
            for function in built.functions
            if isinstance(function, Function)
        )

    def selected(self, kind: Literal["analyze", "schedule"]) -> tuple[str, ...]:
        cases = self.analyze if kind == "analyze" else self.schedule
        return tuple(dict.fromkeys(case.function for case in cases))

    def untested(
        self, kind: Literal["analyze", "schedule"], module: Module | None = None
    ) -> tuple[str, ...]:
        """The model's own functions that no case of *kind* selected."""
        chosen = set(self.selected(kind))
        return tuple(name for name in self.inventory(module) if name not in chosen)

    def function(self, module: Module, case: FunctionCase) -> Function:
        try:
            found = module.lookup(case.function)
        except ValueError as error:
            raise CorpusError(f"model case {self.id!r}: {error}") from None
        if not isinstance(found, Function):
            raise CorpusError(
                f"model case {self.id!r}: {case.function!r} is a "
                f"{type(found).__name__}, not an HIR function"
            )
        return found


__all__ = [
    "CapabilityGate",
    "CorpusError",
    "FunctionCase",
    "MODELS_ROOT",
    "ModelCase",
    "Outcome",
    "ReferenceCase",
    "TargetFixture",
]
