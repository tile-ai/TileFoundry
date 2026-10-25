"""``ops::mma`` on explicitly gathered per-lane fragments.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
import tilefoundry_nvidia.cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tests.nvidia.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia import CudaTarget

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
TIR = Path(__file__).parents[2] / "fixtures" / "tir"


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_handwritten_mma_matches_torch(target: CudaTarget) -> None:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    MmHandwritten = load_authored_ir(f"{TIR / 'mma.py'}:MmHandwritten")
    runtime = tilefoundry.compile(retarget(MmHandwritten, target), target=target)
    runtime(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_mma_matches_torch_matmul(target: CudaTarget) -> None:
    MmaTile = load_authored_ir(f"{OPS / 'mma_tile.py'}:MmaTile")
    rm = tilefoundry.compile(retarget(MmaTile, target), target=target)
    torch.manual_seed(0)
    a = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(128, dtype=torch.float32, device="cuda")
    rm(a, b, c)
    torch.cuda.synchronize()
    expected = a.view(16, 16).float() @ b.view(16, 8).float()
    assert torch.allclose(c.view(16, 8), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_fragment_mma_matches_torch_matmul(target: CudaTarget) -> None:
    """E4M3 operands give the F32 product of the same values."""
    torch.manual_seed(1)
    a = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(32, 8, device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    Fp8MmaTile = load_authored_ir(f"{OPS / 'fp8_mma_tile.py'}:Fp8MmaTile")
    runtime = tilefoundry.compile(retarget(Fp8MmaTile, target), target=target)
    runtime(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_e4m3_gemm_through_shared_memory_equals_the_f32_product(target: CudaTarget) -> None:
    """Integer inputs keep every partial sum exact, so the tile must equal the F32 product."""
    module = retarget(load_authored_ir(f"{TIR / 'e4m3_gemm.py'}:Fp8Gemm"), target)
    verify_module(module)
    runtime = tilefoundry.compile(module, target=target)
    torch.manual_seed(0)
    a = torch.randint(-4, 5, (16, 128), device="cuda").float().to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (8, 128), device="cuda").float().to(torch.float8_e4m3fn)
    c = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime(a, b, c)
    torch.cuda.synchronize()
    assert torch.equal(c, a.float() @ b.float().T)
