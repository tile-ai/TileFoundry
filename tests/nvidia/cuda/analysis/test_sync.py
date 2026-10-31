"""Where the sync family orders shared memory, and what it refuses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import tilefoundry.analysis.sync as sync
from tests.core._corpus import Case, case, check_case, special
from tilefoundry.analysis import AnalysisError
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.sync import SyncPlanMetadata, event_of, region_position
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.tir.sync import SyncBarrier
from tilefoundry.ir.visitor import collect_exprs

FIXTURES = Path(__file__).parents[1] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


def _planned(module):
    """The analysed Function, its plan and its liveness record."""
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory", "sync"))
    function = result.function
    return function, get_metadata(function, SyncPlanMetadata), get_metadata(function, LivenessMetadata)


def _of_kind(owner, kind):
    return [expr for expr in collect_exprs(owner.body) if isinstance(expr, kind)]


def _thread_regions(function):
    return [
        region
        for region in _of_kind(function, MeshRegion)
        if {topology.name for topology in region.mesh.topologies} == {"thread"}
    ]


class SyncCase(Case):
    """A program the sync family plans, and the steps it places."""


@case
class CpAsyncGemm(SyncCase):
    """In each K slice, a barrier before each copy and a commit, wait and barrier before its read.

    The barrier before the copy keeps it behind the last slice's read of the same bytes.
    """

    path: Path = GEMM

    def guard(self, state, problems):
        function, plan, liveness = _planned(state)
        issued = {
            (event_of(liveness, call), 0): type(call.target.op).__name__
            for call in _of_kind(function, Call)
            if isinstance(call.target, ScheduleOp)
        }
        assert [
            (issued.get(position), tuple(step.kind for step in steps))
            for position, steps in plan.steps
        ] == [
            ("CopyAsync", ("barrier",)),
            ("Copy", ("commit", "wait", "barrier")),
            ("CopyAsync", ("barrier",)),
            ("Copy", ("commit", "wait", "barrier")),
        ]
        (threads,) = _thread_regions(function)
        meshes = {step.mesh for _position, steps in plan.steps for step in steps if step.mesh}
        assert meshes == {threads.mesh}


@case
class NestedSubsets(SyncCase):
    """The writer's half orders its own loop; the barrier between halves covers both."""

    path: Path = FIXTURES / "analysis/sync_nested_subsets.py"

    def guard(self, state, problems):
        function, plan, liveness = _planned(state)
        (inner_at, (inner,)), (between_at, (between,)) = plan.steps
        assert (inner.kind, between.kind) == ("barrier", "barrier")
        threads, producer, consumer = sorted(
            _thread_regions(function), key=lambda region: region_position(liveness, region)
        )
        assert region_position(liveness, consumer) == between_at
        assert (inner.mesh, between.mesh) == (producer.mesh, threads.mesh)
        (loop,) = _of_kind(producer, LoopRegion)
        start, backedge = liveness.loop_windows[id(loop)]
        assert start < inner_at[0] <= backedge


@case
class CapturedInsertUpdate(SyncCase):
    """The returned tile may live in the caller's output, so its fill orders before the loop."""

    path: Path = FIXTURES / "schedule/hir/captured_insert_update.py"

    def guard(self, state, problems):
        function, plan, liveness = _planned(state)
        (loop,) = _of_kind(function, LoopRegion)
        assert [step.kind for step in plan.at(region_position(liveness, loop))] == ["barrier"]


@case
class PlainNestedCalls(SyncCase):
    """A call whose callees hold no shared memory is admitted, and needs no step."""

    path: Path = FIXTURES / "analysis/sync_nested_calls.py"
    name: str | None = "Plain"

    def guard(self, state, problems):
        _function, plan, _liveness = _planned(state)
        assert plan.steps == ()


@case
class StagedNestedCalls(SyncCase):
    """The same nesting refuses the call once its innermost leaf holds shared memory."""

    path: Path = FIXTURES / "analysis/sync_nested_calls.py"
    name: str | None = "Staged"

    def guard(self, state, problems):
        with pytest.raises(AnalysisError, match="touches shared memory; call summaries"):
            _planned(state)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("within_scope", lambda mesh, scope: False, "is issued on a mesh not within its scope's"),
        ("classify", lambda mesh: SyncBarrier.GRID, "a scope that no thread mesh binds"),
        ("_LOOP_LIMIT", 1, "did not settle within its limit"),
    ],
    ids=["an issuing mesh outside its scope", "a barrier no thread mesh holds", "a loop that does not settle"],
)
def test_the_family_refuses_what_it_cannot_order(name, value, message):
    with patch.object(sync, name, value), pytest.raises(AnalysisError, match=message):
        _planned(load_authored_ir(str(GEMM)))


CASES = special(SyncCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_sync_case(case: SyncCase) -> None:
    check_case(case)
