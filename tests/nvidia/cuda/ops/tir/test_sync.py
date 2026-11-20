"""Exercise the CUDA barrier forms emitted for ``T.sync``.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.nvidia.cuda.schedule._support import _finalized, _RetainedStatements
from tilefoundry import cli
from tilefoundry.ir.tir import sync
from tilefoundry_nvidia import cuda

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = cuda.Target("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sync_kernel_runs_and_squares() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    SyncSquare = cli.load_authored_ir(str(Path(__file__).parents[2] / "fixtures/tir/sync.py"))
    runtime = tilefoundry.compile(SyncSquare, target=cuda.Target("nvidia.h200_sxm"))
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warp_slices_exchange_values() -> None:
    one = torch.arange(32, dtype=torch.float32, device="cuda").reshape(1, 32)
    two = torch.arange(64, dtype=torch.float32, device="cuda").reshape(2, 32) + 1000
    one_out = torch.full((1, 32), -1.0, device="cuda")
    two_out = torch.full((2, 32), -1.0, device="cuda")
    SyncSlices = cli.load_authored_ir(f"{OPS / 'sync_slices.py'}:SyncSlices")
    runtime = tilefoundry.compile(SyncSlices, target=_CUDA)
    runtime(one, one_out, two, two_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(one_out, one, rtol=0, atol=0)
    torch.testing.assert_close(two_out, two, rtol=0, atol=0)


def test_a_sync_on_the_current_sliced_scope_is_its_own_threads_barrier() -> None:
    """The scope a sliced ``with`` binds is a legal sync mesh: a bar.sync on its 256 threads."""
    module = cli.load_authored_ir(f"{OPS / 'sync_current_slice.py'}:SyncCurrentSlice")
    statements = _RetainedStatements()
    statements.visit(_finalized(module).lookup(module.entry_function().name).body)
    sliced = [
        op.mesh for op in statements.operations if isinstance(op, sync.Sync) and op.mesh.names[0] == "group"
    ]
    assert sliced
    for mesh in sliced:
        scope = sync.sync_scope(mesh)
        assert sync.classify(mesh) is sync.SyncBarrier.BAR_SYNC
        assert (scope.base, scope.count) == (128, 256)

