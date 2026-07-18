"""Tests for the direct public Schedule service contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import tilefoundry.schedule as schedule_api
from tilefoundry.schedule import (
    Schedule,
    ScheduleOptions,
    ScheduleReport,
    ScheduleResult,
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


class _FakeSchedule:
    stage = "cta"

    def solve(self, module, root, options: ScheduleOptions) -> ScheduleResult:
        return ScheduleResult(module=module, report=_report())


def test_public_surface_is_the_direct_service_contract() -> None:
    assert set(schedule_api.__all__) == {
        "Schedule",
        "ScheduleOptions",
        "ScheduleReport",
        "ScheduleResult",
    }
    assert not hasattr(schedule_api, "solve")
    assert not hasattr(schedule_api, "ScheduleError")


def test_schedule_options_are_immutable_values() -> None:
    options = ScheduleOptions(
        timeout_seconds=5.0,
        workers=1,
        random_seed=7,
        debug_dump_dir=Path("debug"),
    )

    assert options == ScheduleOptions(5.0, 1, 7, Path("debug"))
    with pytest.raises(FrozenInstanceError):
        options.workers = 2


def test_schedule_service_is_invoked_directly() -> None:
    service: Schedule = _FakeSchedule()
    module = object()
    root = object()

    result = service.solve(module, root, ScheduleOptions())

    assert service.stage == "cta"
    assert result.module is module
    assert result.report == _report()


def test_schedule_report_and_result_are_immutable_values() -> None:
    report = _report()
    result = ScheduleResult(module=object(), report=report)

    with pytest.raises(FrozenInstanceError):
        report.selected = 70
    with pytest.raises(FrozenInstanceError):
        result.module = object()


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
