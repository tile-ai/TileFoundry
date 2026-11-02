"""Where the CUDA lowering emits the steps the sync analysis attached, and what it needs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, For, MeshScope, Sequential
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.passes.transforms import LoweringError
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import WgmmaCommitGroup, WgmmaFence, WgmmaWaitGroup

FIXTURES = Path(__file__).parents[1] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


def _analyzed(module, analysis=("memory", "sync")):
    return analyze(*check_program(module, module.entry_function()), analysis=analysis)


def _operations(path):
    result = _analyzed(load_authored_ir(str(path)))
    lowered = _RetainedStatements()
    lowered.visit(finalize(result).lookup(result.function.name).body)
    return [type(op) for op in lowered.operations]


def test_each_row_lands_before_the_column_reads():
    """A commit, a wait and a barrier between each thread's row copy and the column reads."""
    operations = _operations(FIXTURES / "passes/sync_row_windows.py")
    start = operations.index(CopyAsync)
    end = operations.index(Copy, start)
    assert operations[start + 1 : end] == [CpAsyncCommit, CpAsyncWait, Sync]


def test_each_lane_reads_its_own_row_without_a_barrier():
    """Each lane writes and reads back only its own row, so its loop gets no barrier."""
    operations = _operations(FIXTURES / "passes/sync_lane_rows.py")
    assert Copy in operations
    assert Sync not in operations


def test_tma_loads_land_before_the_wgmma_reads_them():
    """A TMA load declares no completion, so it is done when it returns: a barrier suffices.

    The barrier at the loop head keeps the next loads behind the last iteration's read.
    """
    result = _analyzed(load_authored_ir(str(FIXTURES / "schedule/hir/wgmma_a_k_major.py")))
    function = _RetainedStatements()
    function.visit(finalize(result).lookup(result.function.name).body)
    (loop,) = function.loops
    body = _RetainedStatements()
    body.visit(loop.body)
    kinds = [
        type(op) for op in body.operations if isinstance(op, (Sync, CopyAsyncTensor, TiledMma))
    ]
    first = kinds.index(TiledMma)
    assert kinds[: first + 1] == [Sync, CopyAsyncTensor, CopyAsyncTensor, Sync, TiledMma]


def test_the_back_edge_copies_follow_a_barrier():
    """The back edge copies each ring stage into its carried slot after one barrier.

    The copies read what a loader thread's TMA wrote and overwrite what the WGMMA read.
    """
    result = _analyzed(load_authored_ir(str(FIXTURES / "passes/sync_carry_copies.py")))
    function = _RetainedStatements()
    function.visit(finalize(result).lookup(result.function.name).body)
    (loop,) = function.loops
    body = _RetainedStatements()
    body.visit(loop.body)
    kinds = [type(op) for op in body.operations if isinstance(op, (Sync, Copy, TiledMma))]
    first = kinds.index(Copy)
    assert kinds[first - 2 :] == [TiledMma, Sync, Copy, Copy]


def test_each_issue_reading_a_from_registers_gets_its_own_fence():
    """The RS WGMMA over K = 64 issues four times, each after its own ``wgmma_fence``.

    The copy of A before it in the loop body waits for the last iteration's group.
    """
    operations = _operations(FIXTURES / "analysis/sync_wgmma_rs.py")
    start = operations.index(WgmmaFence)
    assert operations[start - 3 : start] == [WgmmaCommitGroup, WgmmaWaitGroup, Copy]
    assert operations[start : start + 10] == [WgmmaFence, TiledMma] * 4 + [
        WgmmaCommitGroup,
        WgmmaWaitGroup,
    ]


def test_a_wgmma_completes_where_it_issues_when_its_wait_has_no_place():
    """No scope holding the loader warp and the warpgroup can run the group's wait.

    So the commit and the wait follow the WGMMA, in the region that issues it.
    """
    result = _analyzed(load_authored_ir(str(FIXTURES / "analysis/sync_wgmma_partial.py")))
    pending, issuing = [finalize(result).lookup(result.function.name).body], []
    while pending:
        stmt = pending.pop()
        inner = stmt.body if isinstance(stmt, (MeshScope, For)) else stmt
        if isinstance(inner, Sequential):
            pending.extend(inner.body)
            kinds = [type(s.callable) for s in inner.body if isinstance(s, Evaluate)]
            if isinstance(stmt, MeshScope) and TiledMma in kinds:
                issuing.append(kinds)
    (kinds,) = issuing
    last = max(index for index, kind in enumerate(kinds) if kind is TiledMma)
    assert kinds[last + 1 : last + 3] == [WgmmaCommitGroup, WgmmaWaitGroup]


def test_both_groups_complete_before_the_copy_into_their_bytes():
    """Each group's commit and wait reach the TIR, then the barrier, before the copy."""
    operations = _operations(FIXTURES / "analysis/sync_two_groups.py")
    start = operations.index(TiledMma) + 1
    assert operations[start : start + 6] == [
        CpAsyncCommit,
        WgmmaCommitGroup,
        CpAsyncWait,
        WgmmaWaitGroup,
        Sync,
        Copy,
    ]


def test_lowering_needs_the_sync_analysis():
    result = _analyzed(load_authored_ir(str(GEMM)), analysis=("memory",))
    with pytest.raises(LoweringError, match="needs the result of analyze"):
        finalize(result)
