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
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.passes.transforms import LoweringError
from tilefoundry.schedule import finalize
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma

FIXTURES = Path(__file__).parents[1] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


def _analyzed(module, analysis=("memory", "sync")):
    return analyze(*check_program(module, module.entry_function()), analysis=analysis)


def test_each_row_lands_before_the_column_reads():
    """A commit, a wait and a barrier between each thread's row copy and the column reads."""
    result = _analyzed(load_authored_ir(str(FIXTURES / "passes/sync_row_windows.py")))
    lowered = _RetainedStatements()
    lowered.visit(finalize(result).lookup(result.function.name).body)
    operations = lowered.operations
    start = next(i for i, op in enumerate(operations) if isinstance(op, CopyAsync))
    end = next(i for i, op in enumerate(operations) if i > start and isinstance(op, Copy))
    assert [type(op) for op in operations[start + 1 : end]] == [CpAsyncCommit, CpAsyncWait, Sync]


def test_each_lane_reads_its_own_row_without_a_barrier():
    """Each lane writes and reads back only its own row, so its loop gets no barrier."""
    result = _analyzed(load_authored_ir(str(FIXTURES / "passes/sync_lane_rows.py")))
    lowered = _RetainedStatements()
    lowered.visit(finalize(result).lookup(result.function.name).body)
    assert any(isinstance(op, Copy) for op in lowered.operations)
    assert not any(isinstance(op, Sync) for op in lowered.operations)


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


def test_lowering_needs_the_sync_analysis():
    result = _analyzed(load_authored_ir(str(GEMM)), analysis=("memory",))
    with pytest.raises(LoweringError, match="needs the result of analyze"):
        finalize(result)
