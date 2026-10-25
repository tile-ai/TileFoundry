"""Exercise the CUDA barrier forms emitted for ``T.sync``.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = CudaTarget("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sync_kernel_runs_and_squares() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    SyncSquare = load_authored_ir(str(Path(__file__).parents[2] / "fixtures/tir/sync.py"))
    runtime = tilefoundry.compile(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warp_slices_exchange_every_participants_value() -> None:
    one = torch.arange(32, dtype=torch.float32, device="cuda").reshape(1, 32)
    two = torch.arange(64, dtype=torch.float32, device="cuda").reshape(2, 32) + 1000
    one_out = torch.full((1, 32), -1.0, device="cuda")
    two_out = torch.full((2, 32), -1.0, device="cuda")
    SyncSlices = load_authored_ir(f"{OPS / 'sync_slices.py'}:SyncSlices")
    runtime = tilefoundry.compile(SyncSlices, target=_CUDA)
    runtime(one, one_out, two, two_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(one_out, one, rtol=0, atol=0)
    torch.testing.assert_close(two_out, two, rtol=0, atol=0)
