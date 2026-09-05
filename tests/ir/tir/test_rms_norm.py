"""CUDA RMSNorm authored fixture runtime coverage."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.rmsnorm import TirRmsnorm
from tilefoundry.target import CudaTarget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rmsnorm_matches_torch() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 128, dtype=torch.float32, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight
    runtime = tilefoundry.compile(TirRmsnorm, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x, weight, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
