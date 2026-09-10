"""Shared integration entry point for multiple authored TIR fixtures."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.mma import MmHandwritten
from tests.fixtures.tir.rmsnorm import TirRmsnorm
from tilefoundry.target import CudaTarget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fixture", [TirRmsnorm, MmHandwritten], ids=["rmsnorm", "mma"])
def test_fixture_pipeline_produces_reference(fixture) -> None:
    """Compile and execute more than one fixture through one integration file."""
    runtime = tilefoundry.compile(fixture, target=CudaTarget("nvidia.h200_sxm"))
    if fixture is TirRmsnorm:
        torch.manual_seed(4)
        x = torch.randn(1, 128, device="cuda")
        weight = torch.randn(128, device="cuda")
        out = torch.empty_like(x)
        runtime(x, weight, out)
        expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
    elif fixture is MmHandwritten:
        torch.manual_seed(5)
        a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
        runtime(a, b, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=2e-2, atol=2e-2)
