"""Analyze obtains its complete dependency closure from the Target value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.registry import Analyzer
from tilefoundry.ir.core.module import Module
from tilefoundry.target import Target, register_target


@register_target
@dataclass(frozen=True)
class _AnalysisTarget(Target):
    name: ClassVar[str] = "test.analysis"
    services: tuple[Analyzer, ...] = ()

    def get_analyzer(self, selector: str) -> Analyzer:
        for service in self.services:
            if service.selector == selector:
                return service
        return super().get_analyzer(selector)


def _module(target: Target) -> tuple[Module, object]:
    function, _, _ = build_demo()
    return Module("demo", (function,), function.name, target=target), function


def test_analyze_orders_a_target_selected_dependency_diamond_once() -> None:
    ran: list[str] = []

    def service(selector: str, requires: tuple[str, ...] = ()) -> Analyzer:
        return Analyzer(selector, lambda *_args: ran.append(selector), requires=requires)

    target = _AnalysisTarget(
        (
            service("base"),
            service("left", ("base",)),
            service("right", ("base",)),
            service("top", ("left", "right")),
        )
    )
    module, function = _module(target)

    result = analyze(module, function, analysis="top")

    assert result.executed == ("base", "left", "right", "top")
    assert ran == list(result.executed)


def test_analyze_reports_missing_root_and_missing_dependency_from_the_target() -> None:
    module, function = _module(_AnalysisTarget())
    with pytest.raises(AnalysisError, match="no analyzer for 'missing'"):
        analyze(module, function, analysis="missing")

    target = _AnalysisTarget((Analyzer("root", lambda *_args: None, requires=("lost",)),))
    module, function = _module(target)
    with pytest.raises(AnalysisError, match="'root' depends on 'lost'"):
        analyze(module, function, analysis="root")


def test_analyze_detects_cycles_declared_by_target_services() -> None:
    target = _AnalysisTarget(
        (
            Analyzer("a", lambda *_args: None, requires=("b",)),
            Analyzer("b", lambda *_args: None, requires=("a",)),
        )
    )
    module, function = _module(target)

    with pytest.raises(AnalysisError, match="a -> b -> a"):
        analyze(module, function, analysis="a")
