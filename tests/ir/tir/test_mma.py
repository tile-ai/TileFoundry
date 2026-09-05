"""CUDA MMA authored fixture runtime coverage."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.mma import MmHandwritten
from tilefoundry.target import CudaTarget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_handwritten_mma_matches_torch() -> None:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime = tilefoundry.compile(MmHandwritten, target=CudaTarget("nvidia.h200_sxm"))
    runtime(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=2e-2, atol=2e-2)
