"""The shared vocabulary every model-driven test is written against.

One model is described once, as a `ModelCase`, and every kind of test reads that
one description: the reference run, the analyses, the schedules, the CLI, and the
end-to-end witnesses. Nothing here copies a model into a smaller graph for one
subsystem's convenience, because a result measured on a copy says nothing about
the program a user would actually hand us.

A case is target-free on purpose. The model source states shapes and dtypes; a
`TargetFixture` states the machine. They meet only when a test asks for it, so the
same model can be asked about on more than one target in one run.

`build()` copies the model's prototype every time. That is not caution about
mutation in the abstract -- an analysis attaches its records to the Call objects
it measured, in place, so two tests sharing one built Module do not share a model,
they share each other's results, and the prototype they came from would collect
every run's. Binding a Target with `dataclasses.replace` copies the Module value
but keeps those very Call objects, so it isolates nothing on its own; the fresh
copy has to come first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import pytest

from tilefoundry.ir.core.module import Module, function_selectors, select
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.base import Target

MODELS_ROOT = Path(__file__).parent

Outcome = Literal["PASS", "BLOCKED"]


class CorpusError(Exception):
    """A case describes something the corpus cannot resolve.

    Deliberately not a `ValueError`. A blocked case absorbs one named failure
    type as its expected result, and the complaints raised here -- a block that
    started passing, a block that failed for another reason -- are the two
    things that must never be absorbed. Sharing a base with a plausible
    `expect` would let exactly those be recorded as the expectation.
    """


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

    def hold(
        self,
        run: Callable[[], object],
        *,
        expect: type[BaseException],
        label: str,
    ) -> object:
        """Run *run* and hold it to what this gate claims about it, returning it.

        The value comes back because a caller that gates a computation usually
        has to go on and judge it: running it and discarding the result gates
        only whether it raised, which is a far weaker claim than the caller
        means to make and reads identically at the call site.

        A blocked case returns nothing, because it has nothing to return -- the
        stated failure is re-raised below.

        A blocked case is a strict expectation in both directions, and the
        expectation is the test result rather than a field in a report. The
        stated failure is re-raised so the case is recorded as an expected
        failure by the runner (see `expected_failure`), which is what makes an
        unexpected success visible as one.

        Two failures are not the expectation and must not be recorded as it: a
        case that breaks for another reason is not the limit anybody signed off
        on, and recording it as one hides a second defect behind the first; and
        a block that starts passing means the matrix now describes a system
        nobody has, which only gets corrected if it breaks the build.
        """
        if not self.blocked:
            return run()
        try:
            run()
        except expect as error:
            if self.reason not in str(error):
                raise CorpusError(
                    f"{label} is blocked on {self.reason!r}, but it failed "
                    f"with: {error}"
                ) from error
            raise
        raise CorpusError(
            f"{label} is recorded as blocked on {self.reason!r}, and it "
            "succeeded; the capability matrix is out of date"
        )

    def expected_failure(self, *, expect: type[BaseException]) -> tuple[object, ...]:
        """The marks that make this gate's claim the case's own test result.

        `strict` so an unexpected success is a failure rather than a quiet
        pass, and `raises` so only the stated kind of failure counts: anything
        else -- including this module's own complaint that the reason did not
        match -- stays a plain failure instead of being absorbed as expected.
        """
        if not self.blocked:
            return ()
        return (
            pytest.mark.xfail(reason=self.reason, raises=expect, strict=True),
        )


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
    inputs: Callable[..., object]
    oracle: Callable[..., object]
    entry: str | None = None
    runner: Callable[..., object] | None = None
    problem_sizes: tuple[str, ...] = ()
    gate: CapabilityGate = field(default_factory=CapabilityGate)

    def __post_init__(self) -> None:
        """Exactly one way of saying what runs.

        `entry` names one Function of the model's own Module, which is checkable
        against that Function's parameters. `runner` is for a boundary that is not
        one Function -- a whole decoder is a tree of them, walked by an
        orchestration method -- and then the arity of a single signature is not a
        thing to check. Neither stated leaves the reference describing nothing;
        both stated leaves two answers to what ran.
        """
        if (self.entry is None) == (self.runner is None):
            raise ValueError(
                f"{self.id}: state exactly one of entry (one Function of the "
                f"model's Module) or runner (a boundary that is not one Function)"
            )


@dataclass(frozen=True)
class FunctionCase:
    """One function of one model, selected to be analysed or scheduled.

    `selector` names it root-relative to the case's Module: a bare function name
    for a Module that owns the kernel itself, or a dotted path through the child
    Modules it was reached by. The path is what makes the selected function's own
    execution domain reachable, which is what an analysis measures against.

    `dims` states an extent for each dimension the function was authored to
    leave open, which is how a model written for decode is asked about at one
    context length. A model with no open dimension states none, and that is not
    the same as a model that has one and cannot be asked -- see `SizedCase`.
    """

    id: str
    selector: str
    problem_sizes: tuple[str, ...] = ()
    gate: CapabilityGate = field(default_factory=CapabilityGate)
    topology: str | None = None
    dims: Mapping[str, int] | None = None


@dataclass(frozen=True)
class SizedCase:
    """Whether a model can be asked about at a context length it chooses.

    A separate capability from analysis itself, and separately reportable. A
    model authored as a single fixed shape analyses perfectly well and has no
    context length to state; recording that as a failure of analysis would call
    a working thing broken, and leaving it out would hide that the model is not
    the shape the corpus is moving towards.

    So it is its own row: the gate says whether this model can be asked at a
    size, and the reason says what stops it.
    """

    id: str
    selector: str
    dims: Mapping[str, int]
    #: The largest size each of ``dims`` may be asked at. Stated here rather than
    #: read off the model's config, because it is a question the corpus asks and
    #: not a property the model source declares.
    ceiling: Mapping[str, int] = field(default_factory=dict)
    topology: str | None = None
    gate: CapabilityGate = field(default_factory=CapabilityGate)


@dataclass(frozen=True)
class ModelCase:
    """One Module of one model, described once, for every kind of test.

    `prototype` is the Module the model's own `model.py` defines, reached by an
    ordinary import. It is the single source of truth and is never handed out:
    `build()` copies it, because analysis annotates the IR it is given.

    `id` names this Module's boundary and `model` names the model it belongs to.
    They differ only for a model whose kernels live in more than one Module -- a
    hybrid stack's two token mixers are different kernels, not one kernel
    configured twice, and a Module is the execution domain of the functions it
    owns, so they cannot be one case. The report groups by `model`, so a model
    stays one row however many Modules it took to describe. A model with one
    Module states nothing: `model` defaults to `id`.
    """

    id: str
    prototype: Module
    reference: ReferenceCase | None = None
    analyze: tuple[FunctionCase, ...] = ()
    schedule: tuple[FunctionCase, ...] = ()
    sized: tuple[SizedCase, ...] = ()
    model: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", self.id)

    def build(self) -> Module:
        """A Module nothing else holds a reference to.

        A copy of the prototype is what makes this true. Two builds share no
        Function and no Call, so an analysis that annotates one is invisible to
        the other and to the prototype they both came from.
        """
        if not isinstance(self.prototype, Module):
            raise CorpusError(
                f"model case {self.id!r}: prototype is a "
                f"{type(self.prototype).__name__}, not a Module"
            )
        return self.prototype.cloned()

    def build_for(self, fixture: TargetFixture) -> Module:
        """A fresh Module aimed at one machine, in the one order that isolates."""
        return fixture.bind(self.build())

    def inventory(self, module: Module | None = None) -> tuple[str, ...]:
        """Every HIR function this model really defines, as root-relative
        selectors, in source order.

        Derived from a built Module rather than written down, so a function added
        to the model appears as untested instead of silently escaping the report.
        Recursive, so a kernel that moved into a child Module stays in the count
        instead of dropping out of the report along with it.
        """
        built = self.build() if module is None else module
        return tuple(selector for selector, _ in function_selectors(built))

    def selected(self, kind: Literal["analyze", "schedule"]) -> tuple[str, ...]:
        cases = self.analyze if kind == "analyze" else self.schedule
        return tuple(dict.fromkeys(case.selector for case in cases))

    def untested(
        self, kind: Literal["analyze", "schedule"], module: Module | None = None
    ) -> tuple[str, ...]:
        """The model's own functions that no case of *kind* selected."""
        chosen = set(self.selected(kind))
        return tuple(name for name in self.inventory(module) if name not in chosen)

    def resolve(self, module: Module, selector: str) -> tuple[Module, Function]:
        """What *selector* names below *module*: the Module to measure against,
        and the Function it names.

        The selected Module rather than the outer root, because a function's cost
        is a fact about the execution domain that owns it -- its Target and its
        topology budget -- and for a nested kernel that domain is its own Module.

        The selector must end in a function. A path stopping at a child Module
        would resolve to whatever that Module happens to nominate as its default
        step, which is an answer to a question no case asked.
        """
        *path, name = selector.split(".")
        try:
            owner = select(module, ".".join(path))
            if any(child.name == name for child in owner.modules):
                raise ValueError(
                    f"{selector!r} names the child Module {name!r}, not a "
                    f"function; name the function of it this case selects"
                )
            found = owner.lookup(name)
            selected = select(module, selector)
        except (TypeError, ValueError) as error:
            raise CorpusError(f"model case {self.id!r}: {error}") from None
        if not isinstance(found, Function):
            raise CorpusError(
                f"model case {self.id!r}: {selector!r} is a "
                f"{type(found).__name__}, not an HIR function"
            )
        return selected, found


__all__ = [
    "CapabilityGate",
    "CorpusError",
    "FunctionCase",
    "MODELS_ROOT",
    "ModelCase",
    "Outcome",
    "ReferenceCase",
    "SizedCase",
    "TargetFixture",
]
