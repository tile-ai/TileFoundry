"""Sync cases for where the CUDA lowering emits the attached steps, and what it needs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, GEMM, SyncCase, _analyzed
from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.passes.transforms import LoweringError
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import WgmmaCommitGroup, WgmmaFence, WgmmaWaitGroup


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
        start = operations.index(CopyAsync)
        end = operations.index(Copy, start)
        assert operations[start + 1 : end] == [CpAsyncCommit, CpAsyncWait, Sync]


@case
class LaneRows(SyncCase):
    """Each lane writes and reads back only its own row, so its loop gets no barrier."""

    path: Path = FIXTURES / "passes/sync_lane_rows.py"

    def guard(self, state, problems):
        operations = _operations(state)
        assert Copy in operations
        assert Sync not in operations


@case
class TmaLoadsBeforeWgmma(SyncCase):
    """A TMA load declares no completion, so it is done when it returns: a barrier suffices.

    The barrier at the loop head keeps the next loads behind the last iteration's read.
    """

    path: Path = FIXTURES / "schedule/hir/wgmma_a_k_major.py"

    def guard(self, state, problems):
        kinds = [
            type(op)
            for op in _loop_operations(state)
            if isinstance(op, (Sync, CopyAsyncTensor, TiledMma))
        ]
        first = kinds.index(TiledMma)
        assert kinds[: first + 1] == [Sync, CopyAsyncTensor, CopyAsyncTensor, Sync, TiledMma]


@case
class CarryCopies(SyncCase):
    """The back edge copies each ring stage into its carried slot after one barrier.

    The copies read what a loader thread's TMA wrote and overwrite what the WGMMA read.
    """

    path: Path = FIXTURES / "passes/sync_carry_copies.py"

    def guard(self, state, problems):
        kinds = [
            type(op) for op in _loop_operations(state) if isinstance(op, (Sync, Copy, TiledMma))
        ]
        first = kinds.index(Copy)
        assert kinds[first - 2 :] == [TiledMma, Sync, Copy, Copy]


@case
class WgmmaRegisterOperand(SyncCase):
    """The RS WGMMA over K = 64 issues four times, each after its own ``wgmma_fence``.

    The copy of A before it in the loop body waits for the last iteration's group.
    """

    path: Path = FIXTURES / "analysis/sync_wgmma_rs.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(WgmmaFence)
        assert operations[start - 3 : start] == [WgmmaCommitGroup, WgmmaWaitGroup, Copy]
        assert operations[start : start + 10] == [WgmmaFence, TiledMma] * 4 + [
            WgmmaCommitGroup,
            WgmmaWaitGroup,
        ]


@case
class TwoGroups(SyncCase):
    """Each group's commit and wait reach the TIR, then the barrier, before the copy."""

    path: Path = FIXTURES / "analysis/sync_two_groups.py"

    def guard(self, state, problems):
        operations = _operations(state)
        start = operations.index(TiledMma) + 1
        assert operations[start : start + 6] == [
            CpAsyncCommit,
            WgmmaCommitGroup,
            CpAsyncWait,
            WgmmaWaitGroup,
            Sync,
            Copy,
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
        with pytest.raises(LoweringError, match="needs the result of analyze"):
            finalize(result)
