"""Analyze obtains its complete dependency closure from the Target value."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from tilefoundry import analysis, cli, target
from tilefoundry.ir import core, types
from tilefoundry_nvidia import cuda


@dataclass(frozen=True)
class _AnalysisTarget(target.Target):
    name: ClassVar[str] = "test.analysis"
    services: tuple[target.Analyzer, ...] = ()

    def get_analyzer(self, selector: str) -> target.Analyzer:
        for service in self.services:
            if service.selector == selector:
                return service
        return super().get_analyzer(selector)


RMSNORM = f"{Path(__file__).parents[1] / 'fixtures/placed/rmsnorm.py'}:RmsnormModule"


def _module(tgt: target.Target) -> tuple[core.Module, object]:
    function = cli.load_authored_ir(RMSNORM).entry_function()
    return core.Module("rmsnorm", (function,), function.name, target=tgt), function


def test_analyze_runs_dependency_union_once() -> None:
    ran: list[tuple[str, int]] = []

    def service(selector: str, requires: tuple[str, ...] = ()) -> target.Analyzer:
        return target.Analyzer(
            selector,
            lambda function, _context: ran.append((selector, id(function))),
            requires=requires,
        )

    tgt = _AnalysisTarget(
        (
            service("base"),
            service("left", ("base",)),
            service("right", ("base",)),
            service("top", ("left", "right")),
        )
    )
    module, function = _module(tgt)

    result = analysis.analyze(*analysis.check_program(module, function), analysis=("left", "top", "right", "left"))

    assert result.analyses == ("left", "top", "right")
    assert result.executed == ("base", "left", "right", "top")
    assert [selector for selector, _function in ran] == list(result.executed)
    assert {function for _selector, function in ran} == {id(result.function)}


def test_analyze_reports_missing_analyzers() -> None:
    module, function = _module(_AnalysisTarget())
    with pytest.raises(analysis.Error, match="no analyzer for 'missing'"):
        analysis.analyze(*analysis.check_program(module, function), analysis="missing")

    tgt = _AnalysisTarget((target.Analyzer("root", lambda *_args: None, requires=("lost",)),))
    module, function = _module(tgt)
    with pytest.raises(analysis.Error, match="'root' depends on 'lost'"):
        analysis.analyze(*analysis.check_program(module, function), analysis="root")

    tgt = _AnalysisTarget(
        (
            target.Analyzer(
                "needs-facts",
                lambda _function, context: context.target.get_facts(
                    type("MissingFacts", (), {})
                ),
            ),
        )
    )
    module, function = _module(tgt)
    with pytest.raises(
        analysis.Error,
        match=r"needs-facts: _AnalysisTarget \(test.analysis\): no Facts projection",
    ):
        analysis.analyze(*analysis.check_program(module, function), analysis="needs-facts")


def test_analyze_detects_dependency_cycles() -> None:
    tgt = _AnalysisTarget(
        (
            target.Analyzer("a", lambda *_args: None, requires=("b",)),
            target.Analyzer("b", lambda *_args: None, requires=("a",)),
        )
    )
    module, function = _module(tgt)

    with pytest.raises(analysis.Error, match="a -> b -> a"):
        analysis.analyze(*analysis.check_program(module, function), analysis="a")


def test_analyze_raises_provider_value_errors() -> None:
    @dataclass(frozen=True)
    class _BrokenTarget(target.Target):
        name: ClassVar[str] = "test.broken-analysis"

        def get_analyzer(self, selector: str) -> target.Analyzer:
            raise ValueError("provider analysis failure")

    module, function = _module(_BrokenTarget())
    with pytest.raises(ValueError, match="provider analysis failure"):
        analysis.analyze(*analysis.check_program(module, function), analysis="broken")

    function = cli.load_authored_ir(RMSNORM).entry_function()
    over_limit = core.Module(
        "over-limit",
        (function,),
        function.name,
        target=cuda.Target("nvidia.h200_sxm"),
        topologies=(types.Topology("cta", 1), types.Topology("thread", 1025)),
    )
    with pytest.raises(ValueError, match="1 <= extent <= 1024"):
        analysis.analyze(*analysis.check_program(over_limit, function), analysis="roofline")
