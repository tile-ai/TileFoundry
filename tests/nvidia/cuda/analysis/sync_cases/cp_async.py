"""Sync cases for asynchronous copies and the barriers that order shared memory."""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, GEMM, SyncCase, _planned
from tilefoundry.ir import core, expr
from tilefoundry_nvidia.cuda import analysis


def _kinds(steps):
    return tuple(step.kind for step in steps)


def _of_kind(owner, kind):
    return [node for node in core.collect_exprs(owner.body) if isinstance(node, kind)]


def _thread_regions(function):
    return [
        region
        for region in _of_kind(function, core.MeshRegion)
        if {topology.name for topology in region.mesh.topologies} == {"thread"}
    ]


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
            ("CopyAsync", None, (analysis.StepKind.BARRIER,)),
            ("Copy", None, (analysis.StepKind.COMMIT, analysis.StepKind.WAIT, analysis.StepKind.BARRIER)),
            ("CopyAsync", None, (analysis.StepKind.BARRIER,)),
            ("Copy", None, (analysis.StepKind.COMMIT, analysis.StepKind.WAIT, analysis.StepKind.BARRIER)),
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
        (threads,) = [region for region in regions if _of_kind(region, core.MeshRegion)]
        halves = [region for region in regions if region is not threads]
        (producer,) = [
            region
            for region in halves
            if any(
                getattr(call.type, "storage", None) is expr.StorageKind.SMEM
                for call in _of_kind(region, expr.Call)
            )
        ]
        (consumer,) = [region for region in halves if region is not producer]
        (loop,) = _of_kind(producer, core.LoopRegion)
        by_owner = {id(owner): tagged for owner, tagged in attached}
        ((between_tag, (between,)),) = by_owner.pop(id(consumer))
        ((owner, ((inner_tag, (inner,)),)),) = [
            (owner, tagged) for owner, tagged in attached if id(owner) in by_owner
        ]
        assert (between_tag, inner_tag) == (None, None)
        assert (inner.kind, between.kind) == (analysis.StepKind.BARRIER, analysis.StepKind.BARRIER)
        assert (inner.mesh, between.mesh) == (producer.mesh, threads.mesh)
        assert any(owner is node for node in core.collect_exprs(loop.body))


@case
class CapturedInsertUpdate(SyncCase):
    """The returned tile may live in the caller's output, so its fill orders before the loop."""

    path: Path = FIXTURES / "schedule/hir/captured_insert_update.py"

    def guard(self, state, problems):
        function, _attached = _planned(state)
        (loop,) = _of_kind(function, core.LoopRegion)
        assert _kinds(expr.get_metadata(loop, analysis.SyncMetadata).at()) == (analysis.StepKind.BARRIER,)
