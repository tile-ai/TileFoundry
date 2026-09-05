"""Cover CUDA async-copy validation, emission, and execution.

The family requires gmem-to-smem matching dtypes and nonnegative wait groups.
GPU staging through copy, commit, and wait must reproduce synchronous output.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tests.fixtures.tir.async_sync import AsyncStage
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


def _copy_async_pf(src_ty: TensorType, dst_ty: TensorType) -> PrimFunction:
    src = Var(type=src_ty, name="src")
    dst = Var(type=dst_ty, name="dst")
    return PrimFunction(
        name="fn",
        params=(src, dst),
        body=Sequential(body=(Evaluate(callable=CopyAsync(), args=(src, dst)), Return())),
    )


def _wait_pf(n: int) -> PrimFunction:
    return PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(Evaluate(callable=CpAsyncWait(n=n), args=()), Return())),
    )


REJECTED = [
    pytest.param(
        _copy_async_pf(
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM),
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM),
        ),
        "destination must be smem",
        id="non_smem_destination",
    ),
    pytest.param(
        _copy_async_pf(
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
        ),
        "source must be gmem",
        id="non_gmem_source",
    ),
    pytest.param(
        _copy_async_pf(
            make_tensor_type((128, 4), DType.f16, storage=StorageKind.GMEM),
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
        ),
        "dtype mismatch",
        id="dtype_mismatch",
    ),
    pytest.param(_wait_pf(-1), "non-negative", id="negative_wait"),
]


@pytest.mark.parametrize(("stated", "refusal"), REJECTED)
def test_verify_rejects_what_the_instruction_cannot_do(stated, refusal) -> None:
    with pytest.raises(VerifyError, match=refusal):
        verify_prim_function(stated)


def test_async_copy_emits_cp_async() -> None:
    """Test async copy emits cp async.

    The kernel forwards ``copy_async`` to the runtime entry and emits the
    group fences for commit / wait.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(AsyncStage, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source
    assert "tilefoundry::ops::copy_async(" in src
    assert "cp.async.commit_group;" in src
    assert "cp.async.wait_group %0;" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_copy_stage_matches_input() -> None:
    torch.manual_seed(3)
    source = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    out = torch.empty_like(source)
    runtime = tilefoundry.compile(AsyncStage, target=CudaTarget("nvidia.h200_sxm"))
    runtime(source, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, source, rtol=0, atol=0)
