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
from tilefoundry.analysis.sync import SyncMetadata, SyncPlanMetadata
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.tir.sync import SyncBarrier
from tilefoundry.ir.types import StorageKind
from tilefoundry.ir.visitor import collect_exprs

FIXTURES = Path(__file__).parents[1] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


def _planned(module):
    """The analysed Function, and each node with the steps the family attached to it."""
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory", "sync"))
    function = result.function
    assert get_metadata(function, SyncPlanMetadata) is not None
    attached = {
        id(expr): (expr, record.steps)
        for expr in collect_exprs(function.body)
        if (record := get_metadata(expr, SyncMetadata)) is not None
    }
    return function, list(attached.values())


def _kinds(steps):
    return tuple(step.kind for step in steps)


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
        function, attached = _planned(state)
        assert [
            (type(owner.target.op).__name__, tag, _kinds(steps))
            for owner, tagged in attached
            for tag, steps in tagged
        ] == [
            ("CopyAsync", None, ("barrier",)),
            ("Copy", None, ("commit", "wait", "barrier")),
            ("CopyAsync", None, ("barrier",)),
            ("Copy", None, ("commit", "wait", "barrier")),
        ]
        (threads,) = _thread_regions(function)
        meshes = {
            step.mesh for _owner, tagged in attached for _tag, steps in tagged for step in steps
        }
        assert meshes - {None} == {threads.mesh}


@case
class NestedSubsets(SyncCase):
    """The writer's half orders its own loop; the barrier between halves covers both."""

    path: Path = FIXTURES / "analysis/sync_nested_subsets.py"

    def guard(self, state, problems):
        function, attached = _planned(state)
        regions = _thread_regions(function)
        (threads,) = [region for region in regions if _of_kind(region, MeshRegion)]
        halves = [region for region in regions if region is not threads]
        (producer,) = [
            region
            for region in halves
            if any(
                getattr(call.type, "storage", None) is StorageKind.SMEM
                for call in _of_kind(region, Call)
            )
        ]
        (consumer,) = [region for region in halves if region is not producer]
        (loop,) = _of_kind(producer, LoopRegion)
        by_owner = {id(owner): tagged for owner, tagged in attached}
        ((between_tag, (between,)),) = by_owner.pop(id(consumer))
        ((owner, ((inner_tag, (inner,)),)),) = [
            (owner, tagged) for owner, tagged in attached if id(owner) in by_owner
        ]
        assert (between_tag, inner_tag) == (None, None)
        assert (inner.kind, between.kind) == ("barrier", "barrier")
        assert (inner.mesh, between.mesh) == (producer.mesh, threads.mesh)
        assert any(owner is expr for expr in collect_exprs(loop.body))


@case
class CapturedInsertUpdate(SyncCase):
    """The returned tile may live in the caller's output, so its fill orders before the loop."""

    path: Path = FIXTURES / "schedule/hir/captured_insert_update.py"

    def guard(self, state, problems):
        function, _attached = _planned(state)
        (loop,) = _of_kind(function, LoopRegion)
        assert _kinds(get_metadata(loop, SyncMetadata).at()) == ("barrier",)


@case
class PlainNestedCalls(SyncCase):
    """A call whose callees hold no shared memory is admitted, and needs no step."""

    path: Path = FIXTURES / "analysis/sync_nested_calls.py"
    name: str | None = "Plain"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert attached == []


@case
class StagedNestedCalls(SyncCase):
    """Once the innermost leaf holds shared memory, the call is one action touching any bytes.

    Nothing else in the root touches shared memory, so the call needs no step.
    """

    path: Path = FIXTURES / "analysis/sync_nested_calls.py"
    name: str | None = "Staged"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert attached == []


@case
class TwiceNestedCalls(SyncCase):
    """Two calls whose callees touch shared memory need a barrier between them.

    The calls sit in the CTA region, where no thread mesh can hold that barrier.
    """

    path: Path = FIXTURES / "analysis/sync_nested_calls.py"
    name: str | None = "Twice"

    def guard(self, state, problems):
        with pytest.raises(AnalysisError, match="a scope that no thread mesh binds"):
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
