"""Tests for the stage-agnostic public scheduling contract."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

import pytest

from tilefoundry.schedule import (
    Schedule,
    ScheduleError,
    ScheduleOptions,
    ScheduleReport,
    ScheduleResult,
    solve,
)


def _report() -> ScheduleReport:
    return ScheduleReport(
        root="root",
        target="fake",
        stage="cta",
        status="OPTIMAL",
        objective_name="makespan",
        unit="ns",
        baseline=120,
        selected=80,
        solver_phase="resource_area",
        proven_objectives=("makespan", "reshard_bytes", "resource_area"),
        best_bound=80,
        gap=0.0,
    )


@dataclass
class _FakeSchedule:
    stage: str = "cta"
    calls: list[tuple[object, object, ScheduleOptions]] = field(default_factory=list)

    def solve(self, module, root, options: ScheduleOptions) -> ScheduleResult:
        self.calls.append((module, root, options))
        return ScheduleResult(module=module, report=_report())


@dataclass
class _FakeTarget:
    schedule: _FakeSchedule
    lookups: list[tuple[type, str]] = field(default_factory=list)

    def service(self, interface: type, stage: str):
        self.lookups.append((interface, stage))
        if interface is not Schedule or stage != self.schedule.stage:
            raise LookupError(f"fake target has no {interface.__name__} service for {stage!r}")
        return self.schedule


@dataclass
class _Root:
    name: str
    target: _FakeTarget


@dataclass
class _Module:
    functions: tuple[_Root, ...]


def _fixture():
    schedule = _FakeSchedule()
    target = _FakeTarget(schedule)
    root = _Root("root", target)
    module = _Module((root,))
    return module, root, target, schedule


def test_solve_dispatches_exact_stage_with_default_options() -> None:
    module, root, target, schedule = _fixture()

    result = solve(module, root=root, stage="cta")

    assert target.lookups == [(Schedule, "cta")]
    assert schedule.calls == [(module, root, ScheduleOptions())]
    assert result.module is module
    assert result.report.stage == "cta"


def test_solve_forwards_the_exact_options_instance() -> None:
    module, root, _, schedule = _fixture()
    options = ScheduleOptions(timeout_seconds=5.0, workers=1, random_seed=7)

    solve(module, root=root, stage="cta", options=options)

    assert schedule.calls[0][2] is options


def test_solve_has_no_target_override() -> None:
    assert "target" not in inspect.signature(solve).parameters


def test_solve_requires_root_membership_before_lookup() -> None:
    module, _, target, _ = _fixture()
    other = _Root("other", target)

    with pytest.raises(ScheduleError, match="root 'other' is not one of module.functions"):
        solve(module, root=other, stage="cta")

    assert target.lookups == []


@pytest.mark.parametrize("stage", ["", None, 0])
def test_solve_rejects_invalid_stage_before_lookup(stage) -> None:
    module, root, target, _ = _fixture()

    with pytest.raises(ScheduleError, match="stage must be a non-empty str"):
        solve(module, root=root, stage=stage)

    assert target.lookups == []


def test_solve_requires_explicit_stage() -> None:
    module, root, _, _ = _fixture()

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'stage'"):
        solve(module, root=root)


def test_solve_rejects_invalid_options_before_lookup() -> None:
    module, root, target, _ = _fixture()

    with pytest.raises(ScheduleError, match="options must be ScheduleOptions or None"):
        solve(module, root=root, stage="cta", options=False)

    assert target.lookups == []


def test_solve_does_not_fall_back_when_stage_lookup_fails() -> None:
    module, root, target, schedule = _fixture()

    with pytest.raises(LookupError, match="no Schedule service for 'warp'"):
        solve(module, root=root, stage="warp")

    assert target.lookups == [(Schedule, "warp")]
    assert schedule.calls == []


def test_report_json_contains_only_the_public_summary_fields() -> None:
    expected = {
        "baseline": 120,
        "best_bound": 80,
        "gap": 0.0,
        "objective_name": "makespan",
        "proven_objectives": ["makespan", "reshard_bytes", "resource_area"],
        "root": "root",
        "selected": 80,
        "solver_phase": "resource_area",
        "stage": "cta",
        "status": "OPTIMAL",
        "target": "fake",
        "unit": "ns",
    }

    rendered = _report().to_json()

    assert rendered == json.dumps(expected, sort_keys=True)
    assert json.loads(rendered) == expected


def test_report_markdown_is_stable() -> None:
    assert _report().to_markdown() == "\n".join(
        (
            "| field | value |",
            "| --- | --- |",
            "| root | root |",
            "| target | fake |",
            "| stage | cta |",
            "| status | OPTIMAL |",
            "| objective_name | makespan |",
            "| unit | ns |",
            "| baseline | 120 |",
            "| selected | 80 |",
            "| solver_phase | resource_area |",
            "| proven_objectives | makespan, reshard_bytes, resource_area |",
            "| best_bound | 80 |",
            "| gap | 0.0 |",
        )
    )
