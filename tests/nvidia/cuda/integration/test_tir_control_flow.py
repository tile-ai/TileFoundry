"""End-to-end authored TIR control-flow runtime coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_authored_for_if_square_matches_torch() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cuda")
    expected = x.square()
    TirSquare = load_authored_ir(str(Path(__file__).parents[1] / "fixtures/tir/square.py"))
    runtime = tilefoundry.compile(TirSquare, target=CudaTarget("nvidia.h200_sxm"))
    assert runtime.name == "TirSquare.square_host"
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
