"""Async ``cp.async`` staging ops: ``copy_async`` / ``cp_async_commit`` /
``cp_async_wait``.

No corpus model stages through ``cp.async``, so this file is the whole witness
for the op family. It keeps the parts a passing program cannot show:

- ``CopyAsync`` verify requires an smem destination, a gmem source, and a
  matching dtype; ``CpAsyncWait.n`` must be a non-negative int;
- codegen forwards ``CopyAsync`` to the runtime ``ops::copy_async`` entry and
  emits the group fences for commit / wait;
- on GPU, a ``copy_async -> commit -> wait`` staging of a Split gmem source
  into a full shared tile reproduces the input (matches a synchronous copy).

The three ops' ``UnitType`` results and the accepted verify are not asserted
separately: the module below parses the ``T`` surface, types it and verifies it
on the way to being lowered and run, so an unsound one fails there.
"""
from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

# ── verify ───────────────────────────────────────────────────────────────────


def _copy_async_pf(src_ty: TensorType, dst_ty: TensorType) -> PrimFunction:
    src = Var(type=src_ty, name="src")
    dst = Var(type=dst_ty, name="dst")
    return PrimFunction(
        name="fn",
        params=(src, dst),
        body=Sequential(
            body=(Evaluate(callable=CopyAsync(), args=(src, dst)), Return())
        ),
    )


def _wait_pf(n: int) -> PrimFunction:
    return PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(Evaluate(callable=CpAsyncWait(n=n), args=()), Return())),
    )


#: Every way an async copy can be stated wrong: the hardware instruction moves gmem
#: into smem, in one dtype, and waits on a count of outstanding groups.
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


# ── module for emit + GPU staging ────────────────────────────────────────────


@module(entry="async_stage_host")
class AsyncStage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def async_stage_device(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        with Mesh(Topology("thread", 128), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(
                a,
                layout=ShardLayout(
                    layout=Layout(shape=(128, 4), strides=(4, 1)),
                    attrs=(Split(0),),
                    mesh=m,
                ),
            )
            # A flat CTA-shared staging tile: the whole block sees all 512
            # elements. ``copy_async`` places each thread's row at its flat
            # ``local_offset`` within it.
            s = T.alloc_tensor(
                TensorType(
                    shape=(512,),
                    dtype=DType.f32,
                    layout=ShardLayout(
                        layout=Layout(shape=(512,), strides=(1,)),
                        attrs=(Broadcast(),),
                        mesh=m,
                    ),
                    storage=StorageKind.SMEM,
                )
            )
            T.copy_async(a_view, s)      # each thread stages its row at its offset
            T.cp_async_commit()
            T.cp_async_wait(n=0)         # drain: the full tile has landed
            T.sync(m)
            # The full staged tile is now visible to every thread; write it back
            # to the plain output (the sync-copy reference result).
            T.copy(s, b)

    @prim_func(target=CpuTarget())
    def async_stage_host(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        launch(async_stage_device, a, b, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_async_copy_emits_cp_async() -> None:
    """The kernel forwards ``copy_async`` to the runtime entry and emits the
    group fences for commit / wait."""
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
    """A ``copy_async -> commit -> wait`` staging of a Split gmem source into a
    full shared tile reproduces the input (matches a synchronous copy)."""
    rm = tilefoundry.compile(AsyncStage, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    a = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    rm(a, b)
    torch.cuda.synchronize()
    assert torch.allclose(b, a, rtol=0, atol=0)
