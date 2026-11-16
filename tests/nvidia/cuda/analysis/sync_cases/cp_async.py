"""Sync cases for asynchronous copies and the barriers that order shared memory."""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, GEMM, SyncCase, _planned
from tilefoundry.ir import core, expr, types
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
    """Each K slice gets a barrier before its copy, and a commit, wait and barrier before its read.

    The barrier before the copy keeps it behind the last slice's read of the same bytes.
    The commit and the wait run in the read's region, which holds every thread that
    issued the copy, so each copier waits for its own group before the barrier.
    """

    path: Path = GEMM

    def guard(self, state, problems):
        function, attached = _planned(state)
        regions = _thread_regions(function)
        (threads,) = [region for region in regions if region.mesh.names == ("warp", "lane")]
        copiers = [
            region
            for region in regions
            if region.mesh.names == ("thread",) and not isinstance(region.mesh.layout, types.ComposedLayout)
        ]
        assert [
            ("region" if isinstance(owner, core.MeshRegion) else type(owner.target.op).__name__, tag, _kinds(steps))
            for owner, tagged in attached
            for tag, steps in tagged
        ] == [
            ("region", None, (analysis.StepKind.BARRIER,)),
            ("Copy", None, (analysis.StepKind.COMMIT, analysis.StepKind.WAIT, analysis.StepKind.BARRIER)),
            ("region", None, (analysis.StepKind.BARRIER,)),
            ("Copy", None, (analysis.StepKind.COMMIT, analysis.StepKind.WAIT, analysis.StepKind.BARRIER)),
        ]
        entered = [owner for owner, _ in attached if isinstance(owner, core.MeshRegion)]
        assert len(entered) == len(copiers) and all(a is b for a, b in zip(entered, copiers, strict=True))
        meshes = {
            step.mesh for _owner, tagged in attached for _tag, steps in tagged for step in steps
        }
        assert meshes - {None} == {threads.mesh}
        reads = [owner for owner, _ in attached if not isinstance(owner, core.MeshRegion)]
        inside = {id(node) for node in core.collect_exprs(threads.body)}
        assert all(id(read) in inside for read in reads)
        assert types.size(types.levels(threads.mesh)[0]) == threads.mesh.topologies[0].size


@case
class NestedSubsets(SyncCase):
    """The half that hands a tile over orders its own loop, and a barrier covers both halves."""

    path: Path = FIXTURES / "analysis/sync_nested_subsets.py"

    def guard(self, state, problems):
        function, attached = _planned(state)
        regions = _thread_regions(function)
        (threads,) = [region for region in regions if not isinstance(region.mesh.layout, types.ComposedLayout)]
        producer, consumer = sorted(
            (region for region in regions if types.size(types.levels(region.mesh)[0]) == 32),
            key=lambda region: region.mesh.layout.offset,
        )
        (loop,) = _of_kind(producer, core.LoopRegion)
        by_owner = {id(owner): tagged for owner, tagged in attached}
        ((between_tag, (between,)),) = by_owner.pop(id(consumer))
        assert (between_tag, between.kind, between.mesh) == (None, analysis.StepKind.BARRIER, threads.mesh)
        inner = [(owner, tagged) for owner, tagged in attached if id(owner) in by_owner]
        assert len(inner) == 2
        in_loop = {id(node) for node in core.collect_exprs(loop.body)}
        for owner, tagged in inner:
            ((tag, (step,)),) = tagged
            assert id(owner) in in_loop
            assert (tag, step.kind, step.mesh) == (None, analysis.StepKind.BARRIER, producer.mesh)


@case
class CapturedInsertUpdate(SyncCase):
    """The returned tile may live in the caller's output, so its fill orders before the loop."""

    path: Path = FIXTURES / "schedule/hir/captured_insert_update.py"

    def guard(self, state, problems):
        function, _attached = _planned(state)
        (loop,) = _of_kind(function, core.LoopRegion)
        assert _kinds(expr.get_metadata(loop, analysis.SyncMetadata).at()) == (analysis.StepKind.BARRIER,)
