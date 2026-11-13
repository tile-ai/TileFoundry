"""Sync cases for where the CUDA lowering emits the attached steps, and what it needs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, GEMM, SyncCase, _analyzed
from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry import analysis
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.ir.tir import memory as tir_memory
from tilefoundry_nvidia.cuda.ir.tir import nn, sync


def _lowered(module):
    """The retained statements of the entry the analysed *module* lowers to."""
    result = _analyzed(module)
    lowered = _RetainedStatements()
    lowered.visit(finalize(result).lookup(result.function.name).body)
    return lowered


def _operations(module):
    return [type(op) for op in _lowered(module).operations]


class _Issuers(core.StmtVisitor[None]):
    """Each operation of a lowered body with the mesh of the innermost scope around it."""

    def __init__(self):
        self.meshes = []
        self.issued = []

    def visit_MeshScope(self, stmt, ctx=None):
        self.meshes.append(stmt.mesh)
        self.visit(stmt.body)
        self.meshes.pop()

    def visit_Evaluate(self, stmt, ctx=None):
        self.issued.append((stmt.callable, self.meshes[-1] if self.meshes else None))


def _loop_operations(module):
    """The operations in the body of the lowered entry's only loop."""
    (loop,) = _lowered(module).loops
    body = _RetainedStatements()
    body.visit(loop.body)
    return body.operations


@case
class RowWindows(SyncCase):
    """A commit, a wait and a barrier between each thread's row copy and the column reads."""

    path: Path = FIXTURES / "passes/sync_row_windows.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(tir.async_copy.CopyAsync)
        end = operations.index(memory.Copy, start)
        assert operations[start + 1 : end] == [tir.async_copy.CpAsyncCommit, tir.async_copy.CpAsyncWait, tir.sync.Sync]


@case
class LaneRows(SyncCase):
    """Each lane writes and reads back only its own row, so its loop gets no barrier."""

    path: Path = FIXTURES / "passes/sync_lane_rows.py"

    def guard(self, state, problems):
        operations = _operations(state)
        assert memory.Copy in operations
        assert tir.sync.Sync not in operations


@case
class TmaLoadsBeforeWgmma(SyncCase):
    """A TMA load is done when it returns, so a barrier suffices before the WGMMA reads it.

    The warp publishes its plain copy into the load's global source with a global proxy
    fence and meets before the load. The barrier at the loop head keeps the next loads
    behind the last iteration's read.
    """

    path: Path = FIXTURES / "schedule/hir/wgmma_a_k_major.py"

    def guard(self, state, problems):
        kinds = [
            (type(op), op.storage) if isinstance(op, sync.FenceProxyAsync) else type(op)
            for op in _loop_operations(state)
            if isinstance(op, (tir.sync.Sync, sync.FenceProxyAsync, tir_memory.CopyAsyncTensorTx, nn.TiledMma))
        ]
        first = kinds.index(nn.TiledMma)
        assert kinds[: first + 1] == [
            tir.sync.Sync,
            (sync.FenceProxyAsync, expr.StorageKind.GMEM),
            tir.sync.Sync,
            tir_memory.CopyAsyncTensorTx,
            tir_memory.CopyAsyncTensorTx,
            tir.sync.Sync,
            nn.TiledMma,
        ]


@case
class TmaStoreAfterPlainWrites(SyncCase):
    """A TMA store follows plain writes to both of its ends, each published to the async proxy.

    The kernel zeroes the output first, so a global proxy fence and a barrier follow the fill.
    The epilogue's plain copy into the staging tile takes a shared proxy fence and a barrier
    before the store reads the tile.
    """

    path: Path = FIXTURES / "schedule/hir/gemm_8192x17408x5120_tma_store.py"

    def guard(self, state, problems):
        kinds = [
            (type(op), op.storage) if isinstance(op, sync.FenceProxyAsync) else type(op)
            for op in _lowered(state).operations
            if isinstance(op, (memory.Fill, memory.Copy, sync.FenceProxyAsync, tir.sync.Sync, tir_memory.CopyAsyncTensor))
        ]
        fill = kinds.index(memory.Fill)
        assert kinds[fill : fill + 3] == [memory.Fill, (sync.FenceProxyAsync, expr.StorageKind.GMEM), tir.sync.Sync]
        store = kinds.index(tir_memory.CopyAsyncTensor)
        assert kinds[store - 3 : store] == [memory.Copy, (sync.FenceProxyAsync, None), tir.sync.Sync]


@case
class TmaStoreBetweenReads(SyncCase):
    """Plain accesses to a global tile are published before a TMA store overwrites it.

    They take a global proxy fence and a barrier before the store; the read after
    the store takes a barrier alone.
    """

    path: Path = FIXTURES / "passes/sync_tma_store_between_reads.py"

    def guard(self, state, problems):
        kinds = [
            (type(op), op.storage) if isinstance(op, sync.FenceProxyAsync) else type(op)
            for op in _lowered(state).operations
            if isinstance(op, (memory.Copy, sync.FenceProxyAsync, tir.sync.Sync, tir_memory.CopyAsyncTensor))
        ]
        store = kinds.index(tir_memory.CopyAsyncTensor)
        assert kinds[store - 2 : store + 3] == [
            (sync.FenceProxyAsync, expr.StorageKind.GMEM),
            tir.sync.Sync,
            tir_memory.CopyAsyncTensor,
            tir.sync.Sync,
            memory.Copy,
        ]


@case
class CarryCopies(SyncCase):
    """The back edge copies each ring stage into its carried slot after one barrier.

    The copies read what a loader thread's TMA wrote and overwrite what the WGMMA read.
    """

    path: Path = FIXTURES / "passes/sync_carry_copies.py"

    def guard(self, state, problems):
        kinds = [
            type(op) for op in _loop_operations(state) if isinstance(op, (tir.sync.Sync, memory.Copy, nn.TiledMma))
        ]
        first = kinds.index(memory.Copy)
        assert kinds[first - 2 :] == [nn.TiledMma, tir.sync.Sync, memory.Copy, memory.Copy]


@case
class WgmmaRegisterOperand(SyncCase):
    """The RS WGMMA over K = 64 issues four times, each after its own ``wgmma_fence``.

    The copy of A before it in the loop body waits for the last iteration's group.
    """

    path: Path = FIXTURES / "analysis/sync_wgmma_rs.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(sync.WgmmaFence)
        assert operations[start - 3 : start] == [sync.WgmmaCommitGroup, sync.WgmmaWaitGroup, memory.Copy]
        assert operations[start : start + 10] == [sync.WgmmaFence, nn.TiledMma] * 4 + [
            sync.WgmmaCommitGroup,
            sync.WgmmaWaitGroup,
        ]


@case
class OneThreadCpAsync(SyncCase):
    """One thread's read of its own ``cp.async`` row takes a commit and a wait, and no barrier."""

    path: Path = FIXTURES / "analysis/sync_one_thread_cp_async.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(tir.async_copy.CopyAsync)
        assert operations[start : start + 4] == [
            tir.async_copy.CopyAsync,
            tir.async_copy.CpAsyncCommit,
            tir.async_copy.CpAsyncWait,
            memory.Copy,
        ]
        assert tir.sync.Sync not in operations


@case
class OneWriterRow(SyncCase):
    """One thread lands each row through ``cp.async`` and every thread reads it.

    The commit and the wait run over the whole mesh, which holds the writer, before the barrier.
    """

    path: Path = FIXTURES / "schedule/hir/staged_nested.py"

    def guard(self, state, problems):
        result = _analyzed(state)
        issuers = _Issuers()
        issuers.visit(finalize(result).lookup(result.function.name).body)
        kinds = (
            tir.async_copy.CopyAsync,
            tir.async_copy.CpAsyncCommit,
            tir.async_copy.CpAsyncWait,
            tir.sync.Sync,
            memory.Copy,
        )
        issued = [(type(op), mesh) for op, mesh in issuers.issued if isinstance(op, kinds)]
        start = [kind for kind, _ in issued].index(tir.async_copy.CopyAsync)
        assert [kind for kind, _ in issued[start : start + 5]] == list(kinds)
        (_, writer), (_, commit), (_, wait) = issued[start : start + 3]
        assert isinstance(writer.layout, types.ComposedLayout)
        assert types.size(types.levels(writer)[0]) == 1
        assert commit == wait and types.size(types.levels(wait)[0]) == wait.topologies[0].size


@case
class OneWriterFill(SyncCase):
    """Lowering fills unsplit shared memory from one thread, and the family accepts it for that.

    Both take the writer from ``transfer_meshes``. Without its one-thread slice every
    thread of the half fills the whole tile, and the family refuses the program.
    """

    path: Path = FIXTURES / "schedule/hir/mesh_zeros_submesh.py"

    def guard(self, state, problems):
        result = _analyzed(state)
        issuers = _Issuers()
        issuers.visit(finalize(result).lookup(result.function.name).body)
        (writer,) = [mesh for op, mesh in issuers.issued if isinstance(op, memory.Fill)]
        assert isinstance(writer.layout, types.ComposedLayout)
        assert types.size(types.levels(writer)[0]) == 1
        everyone = patch(
            "tilefoundry.passes.transforms.convert_hir_to_tir.first_thread", lambda enclosing, topologies: None
        )
        with everyone, pytest.raises(analysis.Error, match="two threads that issue Fill tensor write one shared"):
            _analyzed(self.load())


@case
class TwoGroups(SyncCase):
    """Each group's commit and wait reach the TIR, then the barrier, before the copy."""

    path: Path = FIXTURES / "analysis/sync_two_groups.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(nn.TiledMma) + 1
        assert operations[start : start + 6] == [
            tir.async_copy.CpAsyncCommit,
            sync.WgmmaCommitGroup,
            tir.async_copy.CpAsyncWait,
            sync.WgmmaWaitGroup,
            tir.sync.Sync,
            memory.Copy,
        ]


@case
class LoweringNeedsSync(SyncCase):
    """Lowering refuses a program that only the memory analysis has seen.

    ``CpAsyncGemm`` owns the same file, so this case names its module to get its own id.
    """

    path: Path = GEMM
    name: str | None = "SM89_E4M3_GEMM"

    def guard(self, state, problems):
        result = _analyzed(state, analysis=("memory",))
        with pytest.raises(transforms.LoweringError, match="needs the result of analyze"):
            finalize(result)
