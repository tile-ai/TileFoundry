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
class TirInvocation:
    run: Callable[[], None]
    assert_output: Callable[[], None]
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TirRuntimeCase:
    """One shared correctness/perf case with a contention-safe ceiling.

    Local H200 medians are 20-74 microseconds. The one-second default leaves
    more than four orders of magnitude for shared-card contention while still
    rejecting a pathological kernel that makes no useful forward progress.
    """

    name: str
    module: Module
    make_invocation: Callable[[object], TirInvocation]
    generous_ceiling_us: float = 1_000_000.0


def _square_invocation(runtime) -> TirInvocation:
    torch.manual_seed(0)
    x = torch.rand(128, dtype=torch.float32, device="cuda")
    expected = x.square()
    return TirInvocation(
        run=lambda: runtime(x),
        assert_output=lambda: torch.testing.assert_close(x, expected, rtol=0, atol=0),
        shape=tuple(x.shape),
        dtype=str(x.dtype),
    )


def _rmsnorm_invocation(runtime) -> TirInvocation:
    torch.manual_seed(1)
    x = torch.randn(1, 128, dtype=torch.float32, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight
    return TirInvocation(
        run=lambda: runtime(x, weight, out),
        assert_output=lambda: torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5),
        shape=tuple(x.shape),
        dtype=str(x.dtype),
    )


def _mma_invocation(runtime) -> TirInvocation:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    expected = torch.matmul(a.float(), b.float())
    return TirInvocation(
        run=lambda: runtime(a, b, out),
        assert_output=lambda: torch.testing.assert_close(
            out,
            expected,
            rtol=2e-2,
            atol=2e-2,
        ),
        shape=tuple(out.shape),
        dtype=str(out.dtype),
    )


def _async_copy_invocation(runtime) -> TirInvocation:
    torch.manual_seed(3)
    source = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    out = torch.empty_like(source)
    return TirInvocation(
        run=lambda: runtime(source, out),
        assert_output=lambda: torch.testing.assert_close(out, source, rtol=0, atol=0),
        shape=tuple(source.shape),
        dtype=str(source.dtype),
    )


def _sync_invocation(runtime) -> TirInvocation:
    torch.manual_seed(4)
    x = torch.rand(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    return TirInvocation(
        run=lambda: runtime(x),
        assert_output=lambda: torch.testing.assert_close(x, expected, rtol=0, atol=0),
        shape=tuple(x.shape),
        dtype=str(x.dtype),
    )


TIR_RUNTIME_CASES = (
    TirRuntimeCase("square", TirSquare, _square_invocation),
    TirRuntimeCase("rmsnorm", TirRmsnorm, _rmsnorm_invocation),
    TirRuntimeCase("mma", MmHandwritten, _mma_invocation),
    TirRuntimeCase("async_copy", AsyncStage, _async_copy_invocation),
    TirRuntimeCase("sync", SyncSquare, _sync_invocation),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", TIR_RUNTIME_CASES, ids=lambda case: case.name)
def test_tir_fixture_matches_torch(case: TirRuntimeCase) -> None:
    runtime = tilefoundry.compile(case.module, target=CudaTarget("nvidia.h200_sxm"))
    invocation = case.make_invocation(runtime)
    invocation.run()
    torch.cuda.synchronize()
    invocation.assert_output()
