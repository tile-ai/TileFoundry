"""The shared vocabulary every model-driven test is written against.

One model is described once, as a `ModelCase`, and every kind of test reads that
one description: the reference run, the analyses, the CLI, and the
end-to-end witnesses. Nothing here copies a model into a smaller graph for one
subsystem's convenience, because a result measured on a copy says nothing about
the program a user would actually hand us.

A published root states the machine it ships aimed at, and a case names that root.
A `TargetFixture` rebinds the root, so the same model can be asked about on more
than one target in one run; the domain a case measures is selected out of the
bound root and inherits from it.

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

    `bind` overrides whatever the source declares.
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
    """One function of one model, selected to be analysed.

    `selector` names it root-relative to the case's Module: a bare function name
    for a Module that owns the kernel itself, or a dotted path through the child
    Modules it was reached by. The path is what makes the selected function's own
    execution domain reachable, which is what an analysis measures against.

    `dims` states an extent for each dimension the function was authored to
    leave open, which is how a model written for decode is asked about at one
    context length. A model with no open dimension states none, and that is not
    the same as a model that has one and cannot be asked -- see `SizedCase`.

    Whether the function is placed is not stated here. It is a fact about the
    program, asked of the concrete HIR by `states_execution_domain`, so a model
    that gains a placement is asked the larger question without anyone
    remembering to change a flag.
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

    `prototype` is the published root the model's own `model.py` defines, reached
    by an ordinary import. It is the single source of truth and is never handed
    out: `build()` copies it, because analysis annotates the IR it is given.

    `scope` names the domain below that root this case is about, and every
    selector here is relative to it. A case whose domain is the root itself
    states nothing.

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
    sized: tuple[SizedCase, ...] = ()
    model: str = ""
    scope: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", self.id)

    def _root(self) -> Module:
        """A copy of the published root that nothing else holds a reference to."""
        if not isinstance(self.prototype, Module):
            raise CorpusError(
                f"model case {self.id!r}: prototype is a "
                f"{type(self.prototype).__name__}, not a Module"
            )
        return self.prototype.cloned()

    def _scoped(self, root: Module) -> Module:
        """The domain `scope` names below *root*, which inherits *root*'s machine."""
        try:
            return select(root, self.scope)
        except (TypeError, ValueError) as error:
            raise CorpusError(f"model case {self.id!r}: {error}") from None

    def build(self) -> Module:
        """A Module nothing else holds a reference to.

        A copy of the prototype is what makes this true. Two builds share no
        Function and no Call, so an analysis that annotates one is invisible to
        the other and to the prototype they both came from.
        """
        return self._scoped(self._root())

    def build_for(self, fixture: TargetFixture) -> Module:
        """A fresh Module aimed at one machine, in the one order that isolates."""
        return self._scoped(fixture.bind(self._root()))

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

    def selected(self, kind: Literal["analyze"]) -> tuple[str, ...]:
        cases = self.analyze
        return tuple(dict.fromkeys(case.selector for case in cases))

    def untested(
        self, kind: Literal["analyze"], module: Module | None = None
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



@dataclass(frozen=True)
class ConcreteCase:
    """One placed program at one set of extents, ready to be asked a question.

    Built when it is asked for rather than when the inventory is made:
    collecting a parametrized test would otherwise specialise every root in the
    directory before running anything.
    """

    id: str
    dims: Mapping[str, int] | None
    build: Callable[[], tuple[Module, Function]]

    def program(self) -> tuple[Module, Function]:
        return self.build()


def states_execution_domain(
    owner: Module,
    function: Function,
    dims: "Mapping[str, int] | None",
    level: str = "cta",
) -> bool:
    """Whether *function*, once concrete, runs anything inside a Mesh of *level*."""
    from tilefoundry.analysis.check import _resolve_program_geometry
    from tilefoundry.ir.visitor import collect_exprs
    from tilefoundry.ir.core import Call, get_metadata
    from tilefoundry.ir.core.metadata import ExecutionDomainMetadata
    from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

    _module, concrete = _resolve_program_geometry(
        owner, function, dims, TypeInferContext(scope=FunctionScope(owner, function))
    )
    return any(
        (get_metadata(item, ExecutionDomainMetadata) or ExecutionDomainMetadata())
        .at(level)
        is not None
        for item in collect_exprs(concrete.body)
        if isinstance(item, Call)
    )


#: The machine a root that states none is asked on, the way `TargetFixture` binds
#: one. A program with no Target is not yet a question about a machine.
_UNBOUND_MACHINE = "nvidia.h200_sxm"

#: The sizes a placed root is asked at when the sizes themselves are the subject.
#: A selector absent from here is asked at one size its own declaration admits,
#: so adding a fixture never means editing this. A selector present here is one
#: whose separate prefill and decode surfaces each have to be asked about, which
#: is a thing about the program rather than about its declared ranges.
STATED_DIMS: "Mapping[tuple[str, str], tuple[Mapping[str, int], ...]]" = {
    ("qwen3_1_7b_pd", "model"): (
        {"ctx_len": 0, "seq": 512},
        {"ctx_len": 512, "seq": 512},
        {"ctx_len": 512, "seq": 1},
        {"ctx_len": 4608, "seq": 1},
    ),
}

#: One size for a dimension a program leaves open, when no size was stated.
_PROBE = 128


def _probe_dims(owner: Module, function: Function) -> "Mapping[str, int] | None":
    """One concrete extent per open dimension, from its own declared range."""
    from tilefoundry.analysis.check import _program_dim_vars

    declared = _program_dim_vars(owner, function)
    if not declared:
        return None
    return {
        name: max(int(var.lo), min(_PROBE, int(var.hi) - 1))
        for name, var in sorted(declared.items())
    }


def placed_fixture_roots() -> tuple[tuple[str, str, Module], ...]:
    """Every reusable placed program, by the file and the name it is published under.

    Walked rather than listed, so a fixture added to the package joins the
    inventory instead of quietly escaping it. A Module reached as somebody's
    child is not a root: it is asked about through the parent whose machine it
    inherits. One of these files names its neighbour the way a script does,
    because the CLI loads them by path, so the package directory is on the path
    while they import.
    """
    import importlib
    import pkgutil
    import sys

    from tilefoundry.ir.core.module import subtree

    import tests.fixtures.placed as placed

    found: list[tuple[str, str, Module]] = []
    sys.path.insert(0, str(Path(placed.__path__[0])))
    try:
        for info in sorted(pkgutil.iter_modules(placed.__path__), key=lambda i: i.name):
            source = importlib.import_module(f"{placed.__name__}.{info.name}")
            named = [
                (name, value)
                for name in dir(source)
                if not name.startswith("_")
                and isinstance(value := getattr(source, name), Module)
            ]
            children = {
                id(child)
                for _name, owner in named
                for child in subtree(owner)
                if child is not owner
            }
            found.extend(
                (info.name, name, value)
                for name, value in named
                if id(value) not in children
            )
    finally:
        sys.path.pop(0)
    return tuple(found)


def placed_cases(level: str = "cta") -> tuple[ConcreteCase, ...]:
    """Every placed root that answers for *level*, at every size it is asked at.

    A root whose own machine does not run *level* at all is left out here and
    named by the inventory guard instead: a CPU program is not a CTA program
    that failed.
    """
    from tilefoundry.target import CudaTarget

    cases: list[ConcreteCase] = []
    for file, name, published in placed_fixture_roots():
        root = published
        try:
            root.resolve_target()
        except Exception:  # noqa: BLE001 -- a root that names no machine
            root = replace(root, target=CudaTarget(_UNBOUND_MACHINE))
        if level not in root.resolve_target().topology_levels:
            continue
        for selector, function in function_selectors(root):
            owner = select(root, selector)
            for dims in STATED_DIMS.get((file, selector)) or (
                _probe_dims(owner, function),
            ):
                shown = ",".join(f"{k}={v}" for k, v in (dims or {}).items()) or "static"
                cases.append(
                    ConcreteCase(
                        f"{file}.{name}.{selector}[{shown}]",
                        dims,
                        lambda owner=owner, function=function: (owner, function),
                    )
                )
    return tuple(cases)


__all__ = [
    "CapabilityGate",
    "ConcreteCase",
    "CorpusError",
    "FunctionCase",
    "MODELS_ROOT",
    "ModelCase",
    "Outcome",
    "ReferenceCase",
    "STATED_DIMS",
    "SizedCase",
    "TargetFixture",
    "placed_cases",
    "placed_fixture_roots",
    "states_execution_domain",
]
