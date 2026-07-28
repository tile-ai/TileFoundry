"""Composition of the public Analyze operation: closure, ordering, ownership,
and what happens when any of it is declared wrongly.

The analyses here are local to the test. They exercise the composition engine
rather than any real cost model, so the shape of a production analysis family
stays free to be decided where it is defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry.analysis import api
from tilefoundry.analysis.api import AnalysisResult, analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.registry import (
    ANALYSES,
    AnalysisAlgorithm,
    register_analysis,
)
from tilefoundry.analysis.walk import attach, detach, postorder
from tilefoundry.ir.core import IRMetadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.target import CpuTarget
from tilefoundry.target.cuda import CudaTarget


@dataclass(frozen=True)
class _Alpha(IRMetadata):
    value: int = 0


@dataclass(frozen=True)
class _Beta(IRMetadata):
    value: int = 0


def _first_expr(function):
    return next(iter(postorder(function.body)))


def _write(function, metadata) -> None:
    attach(_first_expr(function), metadata)


def _drop(function, kind) -> None:
    """Remove every record of *kind*, whoever wrote it."""
    expr = _first_expr(function)
    kept = tuple(item for item in expr.metadata if not isinstance(item, kind))
    object.__setattr__(expr, "metadata", kept)


def _module() -> tuple[Module, object]:
    """A CUDA module holding one demo function."""
    function, _, _ = build_demo()
    return Module("demo", (function,), function.name, target=CudaTarget()), function


@pytest.fixture
def registered():
    """Register analyses for the test and remove them afterwards.

    There is deliberately no public unregister: a registration is a
    process-level install, so only the pairs a test created are undone.
    """
    created: list[tuple[type, str]] = []

    def register(selector, *, requires=(), produces=(), run=None, target=CudaTarget):
        register_analysis(target, selector, requires=requires, produces=produces)(
            run or (lambda module, function, resolved, options: None)
        )
        created.append((target, selector))
        return selector

    yield register

    for key in created:
        ANALYSES._table.pop(key, None)


def test_a_dependency_diamond_runs_each_analysis_once_in_order(registered) -> None:
    """AC-0-1. A shared dependency is computed once for the whole call, and the
    reported order is the order it ran."""
    ran: list[str] = []

    def record(name):
        return lambda module, function, target, options: ran.append(name)

    registered("d.base", run=record("d.base"))
    registered("d.left", requires=("d.base",), run=record("d.left"))
    registered("d.right", requires=("d.base",), run=record("d.right"))
    registered("d.top", requires=("d.left", "d.right"), run=record("d.top"))

    module, function = _module()
    result = analyze(module, function, analysis="d.top")

    assert isinstance(result, AnalysisResult)
    assert result.analysis == "d.top"
    assert result.executed == ("d.base", "d.left", "d.right", "d.top")
    assert ran == list(result.executed)
    assert ran.count("d.base") == 1
    # Dependencies precede their dependants.
    for dependant, dependency in (("d.left", "d.base"), ("d.top", "d.right")):
        assert result.executed.index(dependency) < result.executed.index(dependant)


def test_an_unregistered_root_and_an_unregistered_dependency_read_differently(
    registered,
) -> None:
    """AC-0-2. A caller's own typo and a broken registration are different
    problems, so they do not produce the same message."""
    module, function = _module()

    with pytest.raises(AnalysisError, match="no 'missing.root' registered"):
        analyze(module, function, analysis="missing.root")

    registered("dep.root", requires=("dep.absent",))
    with pytest.raises(AnalysisError, match="depends on 'dep.absent'"):
        analyze(module, function, analysis="dep.root")


def test_a_dependency_cycle_names_the_cycle(registered) -> None:
    """AC-0-2. A cycle cannot be ordered, so it is reported as the path that
    closes rather than as a stack overflow."""
    registered("c.a", requires=("c.b",))
    registered("c.b", requires=("c.a",))

    module, function = _module()
    with pytest.raises(AnalysisError, match=r"cycle: c\.a -> c\.b -> c\.a"):
        analyze(module, function, analysis="c.a")


def test_every_way_of_touching_another_analysis_record_is_a_violation(
    registered,
) -> None:
    """AC-0-2. Ownership is checked against what reached the IR, not against what
    an analysis says it did, so writing directly cannot bypass it.

    Three shapes, one rule. Writing an undeclared type is the plain case. Writing
    a value *equal* to the one already there is still an overwrite, which is why
    ownership is tracked by identity rather than by value. And removing another
    analysis's record changes the IR as much as overwriting it -- a check that
    only inspected the final state would miss that one, because the entry is
    simply gone.
    """
    module, function = _module()

    registered(
        "own.honest",
        produces=(_Alpha,),
        run=lambda module, function, target, options: _write(function, _Alpha(1)),
    )
    assert analyze(module, function, analysis="own.honest").metadata_types == (_Alpha,)

    registered(
        "own.trespass",
        produces=(_Alpha,),
        run=lambda module, function, target, options: _write(function, _Beta(1)),
    )
    with pytest.raises(AnalysisError, match=r"does not declare: \['_Beta'\]"):
        analyze(module, function, analysis="own.trespass")

    registered(
        "eq.base",
        produces=(_Alpha,),
        run=lambda module, function, target, options: _write(function, _Alpha(7)),
    )
    registered(
        "eq.thief",
        requires=("eq.base",),
        produces=(_Beta,),
        run=lambda module, function, target, options: _write(function, _Alpha(7)),
    )
    with pytest.raises(AnalysisError, match=r"does not declare: \['_Alpha'\]"):
        analyze(module, function, analysis="eq.thief")

    registered(
        "del.vandal",
        requires=("eq.base",),
        produces=(_Beta,),
        run=lambda module, function, target, options: _drop(function, _Alpha),
    )
    with pytest.raises(AnalysisError, match=r"does not declare: \['_Alpha'\]"):
        analyze(module, function, analysis="del.vandal")


def test_a_whole_function_record_is_owned_like_any_other(registered) -> None:
    """A whole-function record hangs on the Function, which is a value the walk
    must reach: otherwise ownership never sees it and no reader is told it is
    there. The Function is also not a place ownership stops being enforced --
    writing an undeclared type to it, or removing a record on it that belongs to
    another analysis, are the same violations they are anywhere else.
    """
    module, function = _module()

    registered(
        "fn.owner",
        produces=(_Alpha,),
        run=lambda mod, fn, target, options: attach(fn, _Alpha(1)),
    )
    result = analyze(module, function, analysis="fn.owner")
    assert get_metadata(function, _Alpha) == _Alpha(1)
    assert result.metadata_types == (_Alpha,)

    registered(
        "fn.trespass",
        produces=(_Alpha,),
        run=lambda mod, fn, target, options: attach(fn, _Beta(1)),
    )
    with pytest.raises(AnalysisError, match=r"does not declare: \['_Beta'\]"):
        analyze(module, function, analysis="fn.trespass")

    registered(
        "fn.eraser",
        requires=("fn.owner",),
        produces=(_Beta,),
        run=lambda mod, fn, target, options: detach(fn, _Alpha),
    )
    with pytest.raises(AnalysisError, match=r"does not declare: \['_Alpha'\]"):
        analyze(module, function, analysis="fn.eraser")


def test_a_failed_preflight_stops_every_analysis(registered, monkeypatch) -> None:
    """AC-0-3. An analysis reads inferred types and assumes a verified
    function, so none may run once either preflight has rejected the IR."""
    ran: list[str] = []
    registered(
        "pf.root", run=lambda module, function, target, options: ran.append("pf.root")
    )

    module, function = _module()
    broken = Module("broken", (function,), function.name, target=CudaTarget())

    def _boom(*_args, **_kwargs):
        raise AnalysisError("preflight rejected this function")

    monkeypatch.setattr(api, "_preflight", _boom)
    with pytest.raises(AnalysisError, match="preflight rejected"):
        analyze(broken, function, analysis="pf.root")

    assert ran == []


def test_a_second_call_recomputes_and_replaces_only_its_own_metadata(
    registered,
) -> None:
    """AC-0-4. Re-running recomputes the closure and refreshes what the closure
    owns, leaving Metadata that belongs to nobody in it alone."""
    counter = {"n": 0}

    def bump(module, function, target, options):
        counter["n"] += 1
        _write(function, _Alpha(counter["n"]))

    registered("re.root", produces=(_Alpha,), run=bump)

    module, function = _module()
    _write(function, _Beta(99))

    first = analyze(module, function, analysis="re.root")
    assert get_metadata(_first_expr(function), _Alpha).value == 1

    second = analyze(module, function, analysis="re.root")
    assert second.executed == first.executed
    assert get_metadata(_first_expr(function), _Alpha).value == 2
    # Metadata outside the closure's ownership is untouched by either call.
    assert get_metadata(_first_expr(function), _Beta).value == 99


def test_dispatch_is_exact_with_no_subclass_or_default_target_fallback(
    registered,
) -> None:
    """AC-0-5. A target-independent analysis is registered per target, so an
    unregistered target is a clear miss rather than a silent default."""
    registered("exact.only")
    module, function = _module()

    class _TunedCuda(CudaTarget):
        """A distinct target that happens to share CudaTarget's base."""

    on_subclass = Module(
        "sub", (function,), function.name, target=_TunedCuda()
    )
    with pytest.raises(AnalysisError, match="_TunedCuda"):
        analyze(on_subclass, function, analysis="exact.only")

    on_cpu = Module("cpu", (function,), function.name, target=CpuTarget())
    with pytest.raises(AnalysisError, match="CpuTarget"):
        analyze(on_cpu, function, analysis="exact.only")


def test_an_algorithm_declaration_is_checked_when_it_is_registered() -> None:
    """A malformed declaration is a registration-time error: it cannot depend
    on itself, repeat a dependency, or claim a type that is not Metadata."""
    with pytest.raises(ValueError, match="cannot require itself"):
        AnalysisAlgorithm(selector="s", run=lambda *a: None, requires=("s",))
    with pytest.raises(ValueError, match="duplicate entries in requires"):
        AnalysisAlgorithm(selector="s", run=lambda *a: None, requires=("a", "a"))
    with pytest.raises(ValueError, match="IRMetadata subclasses"):
        AnalysisAlgorithm(selector="s", run=lambda *a: None, produces=(int,))
    with pytest.raises(ValueError, match="produced twice"):
        AnalysisAlgorithm(
            selector="s", run=lambda *a: None, produces=(_Alpha, _Alpha)
        )
