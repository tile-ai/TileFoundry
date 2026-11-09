"""Sync cases for where the CUDA lowering emits the attached steps, and what it needs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, GEMM, SyncCase, _analyzed
from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry.ir import expr, tir
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
