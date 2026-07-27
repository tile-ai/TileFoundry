"""What the replaced scheduling architecture must no longer contain.

These are structural assertions about the repository rather than about one
behaviour. A superseded mechanism that survives as an unused import, an unused
export, or a compatibility shim is not visible from any behavioural test, and the
next reader cannot tell it from a live one.
"""

from __future__ import annotations

import pathlib

import pytest

import tilefoundry.schedule as schedule_package
import tilefoundry.target as target_package
import tilefoundry.target.amx as amx_package
import tilefoundry.target.cuda as cuda_package
from tilefoundry.registry import UnknownAlgorithmError
from tilefoundry.schedule.registry import SCHEDULES
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.base import Target

_SOURCE = pathlib.Path(target_package.__file__).parent.parent
_RETIRED_NAMES = (
    "bind_services",
    "ScheduleReport",
    "TileStoreFacts",
    "AtomCandidateFacts",
    "AtomCandidateQuery",
    "select_atoms",
    "build_planning_problem",
    "solve_planning_problem",
    "materialize_planning_solution",
    "project_schedule_report",
    "write_debug_dumps",
)


def _sources() -> tuple[pathlib.Path, ...]:
    return tuple(sorted(_SOURCE.rglob("*.py")))


def test_no_retired_scheduling_name_survives_anywhere_in_the_source() -> None:
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for name in _RETIRED_NAMES:
            assert name not in text, f"{path.name} still mentions {name}"


def test_a_target_carries_no_service_lookup() -> None:
    """A target answers by projecting facts, and in no other way."""
    for target in (CudaTarget(), AmxTarget(), CpuTarget(), Target("test")):
        assert not hasattr(target, "service")
        assert not hasattr(target, "_services")
        assert not hasattr(target, "schedule")
    assert "_services" not in {
        field for field in Target.__dataclass_fields__
    }


def test_two_equal_targets_stay_interchangeable() -> None:
    """Nothing is registered against a target value, so equality is total."""
    assert CudaTarget() == CudaTarget()
    assert hash(CudaTarget()) == hash(CudaTarget())
    assert AmxTarget() == AmxTarget()


def test_both_cuda_levels_answer_through_one_public_api() -> None:
    """Two algorithms, one operation, and no selector to choose between them."""
    assert SCHEDULES.selectors_for(CudaTarget) == ("cta", "thread")
    assert SCHEDULES.selectors_for(AmxTarget) == ("core",)
    assert SCHEDULES.selectors_for(CpuTarget) == ()

    partition = SCHEDULES.resolve(CudaTarget(), "cta")
    pipeline = SCHEDULES.resolve(CudaTarget(), "thread")
    assert partition is not pipeline
    with pytest.raises(UnknownAlgorithmError):
        SCHEDULES.resolve(CudaTarget(), "warp")


def test_the_public_schedule_surface_is_the_operation_and_its_results() -> None:
    assert set(schedule_package.__all__) == {
        "PlanVerificationError",
        "ScheduleError",
        "ScheduleOptions",
        "SchedulePlan",
        "ScheduleResult",
        "schedule",
    }


def test_a_backend_package_exports_only_its_hardware_values() -> None:
    """Registration is an import side effect, never an exported name."""
    assert set(cuda_package.__all__) == {"CudaTarget", "H200SXM", "SM90"}
    assert set(amx_package.__all__) == {"AmxTarget", "AppleAmx", "AppleM2Pro"}
