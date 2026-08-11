"""GPU end-to-end for ``tir.cuda.nn.Mma``.

Single SM80 16x8x16 BF16 mma atom: ``c = a @ b`` with ``a`` shape
(M=16, K=16) bf16, ``b`` shape (K=16, N=8) bf16, ``c`` shape
(M=16, N=8) f32. Numerical match against ``torch.matmul(a.float(),
b.float())`` within bf16 tolerance.
"""

from __future__ import annotations

import torch

import tilefoundry
from tests.fixtures.placed.mma_tile import MatmulModule
from tilefoundry.target import CudaTarget


def test_mma_sm80_16x8x16_bf16_matches_torch_matmul() -> None:
    rm = tilefoundry.compile(MatmulModule, target=CudaTarget("nvidia.h200_sxm"))
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    rm(a, b, out)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())

    assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2)
