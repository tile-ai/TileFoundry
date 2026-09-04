"""H200 runtime coverage for committed hand-authored TIR fixtures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.async_sync import AsyncStage, SyncSquare
from tests.fixtures.tir.mma import MmHandwritten
from tests.fixtures.tir.rmsnorm import TirRmsnorm
from tests.fixtures.tir.square import TirSquare
from tilefoundry.ir.core.module import Module
from tilefoundry.target import CudaTarget


@dataclass(frozen=True)
class TirRuntimeCase:
    name: str
    module: Module
    assert_matches: Callable[[object], None]


def _square_matches(runtime) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cuda")
    expected = x.square()
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


def _rmsnorm_matches(runtime) -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 128, dtype=torch.float32, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    runtime(x, weight, out)
    torch.cuda.synchronize()
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)


def _mma_matches(runtime) -> None:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out,
        torch.matmul(a.float(), b.float()),
        rtol=2e-2,
        atol=2e-2,
    )


def _async_copy_matches(runtime) -> None:
    torch.manual_seed(3)
    source = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    out = torch.empty_like(source)
    runtime(source, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, source, rtol=0, atol=0)


def _sync_matches(runtime) -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


TIR_RUNTIME_CASES = (
    TirRuntimeCase("square", TirSquare, _square_matches),
    TirRuntimeCase("rmsnorm", TirRmsnorm, _rmsnorm_matches),
    TirRuntimeCase("mma", MmHandwritten, _mma_matches),
    TirRuntimeCase("async_copy", AsyncStage, _async_copy_matches),
    TirRuntimeCase("sync", SyncSquare, _sync_matches),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", TIR_RUNTIME_CASES, ids=lambda case: case.name)
def test_tir_fixture_matches_torch(case: TirRuntimeCase) -> None:
    runtime = tilefoundry.compile(case.module, target=CudaTarget("nvidia.h200_sxm"))
    case.assert_matches(runtime)
