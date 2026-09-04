"""Contention-tolerant performance smoke coverage for TIR runtime fixtures."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass

import pytest
import torch

import tilefoundry
from tests.integration.test_tir_runtime import (
    TIR_RUNTIME_CASES,
    TirInvocation,
    TirRuntimeCase,
)
from tilefoundry.target import CudaTarget


@dataclass(frozen=True)
class TirTimingResult:
    case_name: str
    shape: tuple[int, ...]
    dtype: str
    median_us: float
    samples_us: tuple[float, ...]


def time_cuda_invocation(
    case_name: str,
    invocation: TirInvocation,
    *,
    warmups: int = 3,
    repeats: int = 9,
) -> TirTimingResult:
    """Time one already-validated invocation with CUDA device events."""
    for _ in range(warmups):
        invocation.run()
    torch.cuda.synchronize()

    event_pairs = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invocation.run()
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize()

    samples_us = tuple(start.elapsed_time(end) * 1_000.0 for start, end in event_pairs)
    if not samples_us or any(not math.isfinite(value) or value <= 0 for value in samples_us):
        raise AssertionError(f"invalid CUDA timing samples for {case_name}: {samples_us!r}")
    return TirTimingResult(
        case_name=case_name,
        shape=invocation.shape,
        dtype=invocation.dtype,
        median_us=statistics.median(samples_us),
        samples_us=samples_us,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", TIR_RUNTIME_CASES, ids=lambda case: case.name)
def test_tir_perf_smoke(case: TirRuntimeCase, record_property) -> None:
    runtime = tilefoundry.compile(case.module, target=CudaTarget("nvidia.h200_sxm"))
    invocation = case.make_invocation(runtime)

    invocation.run()
    torch.cuda.synchronize()
    invocation.assert_output()

    result = time_cuda_invocation(case.name, invocation)
    record_property("tir_timing", json.dumps(asdict(result)))

    assert result.median_us > 0
    assert result.median_us < case.generous_ceiling_us
