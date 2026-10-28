"""End-to-end authored TIR control-flow runtime coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import tvm_ffi

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("extra_host", [False, True], ids=["single-host", "multiple-hosts"])
def test_for_if_square_matches_torch(extra_host) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cuda")
    expected = x.square()
    TirSquare = load_authored_ir(str(Path(__file__).parents[1] / "fixtures/tir/square.py"))
    if extra_host:
        secondary = replace(TirSquare.entry_function(), name="secondary_host")
        TirSquare.functions = (secondary, *TirSquare.functions)
    runtime = tilefoundry.compile(TirSquare, target=CudaTarget("nvidia.h200_sxm"))
    assert runtime.name == "TirSquare.square_host"
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    if extra_host:
        second = torch.randn_like(x)
        expected_second = second.square()
        tvm_ffi.get_global_func("TirSquare.secondary_host")(second)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
