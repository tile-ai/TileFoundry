"""Async ``cp.async`` staging ops: ``copy_async`` / ``cp_async_commit`` /
``cp_async_wait``.

Covers the contract:
- the three ops type-infer to ``UnitType`` and parse via the ``T`` surface;
- ``CopyAsync`` verify requires an smem destination, a gmem source, and a
  matching dtype; ``CpAsyncWait.n`` must be a non-negative int;
- codegen forwards ``CopyAsync`` to the runtime ``ops::copy_async`` entry and
  emits the group fences for commit / wait;
- on GPU, a ``copy_async -> commit -> wait`` staging of a Split gmem source
  into a full shared tile reproduces the input (matches a synchronous copy).
"""
from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tests.ops.typeinfer_utils import infer_call
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType, UnitType, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.ir.types.storage import StorageKind

# ── typeinfer ────────────────────────────────────────────────────────────────


def test_copy_async_typeinfers_to_unit() -> None:
    gmem = make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM)
    smem = make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM)
    assert isinstance(infer_call(CopyAsync(), gmem, smem), UnitType)


def test_fence_ops_typeinfer_to_unit() -> None:
    # The commit / wait fences carry no operands; call the registered typeinfer
    # handler directly (the operand-driven harness needs at least one input).
    from tilefoundry.visitor_registry import typeinfer_registry  # noqa: PLC0415

    assert isinstance(typeinfer_registry.lookup(CpAsyncCommit)(None, None), UnitType)
    assert isinstance(typeinfer_registry.lookup(CpAsyncWait)(None, None), UnitType)


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


def test_verify_accepts_gmem_to_smem_same_dtype() -> None:
    verify_prim_function(
        _copy_async_pf(
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM),
            make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
        )
    )


def test_verify_rejects_non_smem_destination() -> None:
    with pytest.raises(VerifyError, match="destination must be smem"):
        verify_prim_function(
            _copy_async_pf(
                make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM),
                make_tensor_type((128, 4), DType.f32, storage=StorageKind.GMEM),
            )
        )


def test_verify_rejects_non_gmem_source() -> None:
    with pytest.raises(VerifyError, match="source must be gmem"):
        verify_prim_function(
            _copy_async_pf(
                make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
                make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
            )
        )


def test_verify_rejects_dtype_mismatch() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch"):
        verify_prim_function(
            _copy_async_pf(
                make_tensor_type((128, 4), DType.f16, storage=StorageKind.GMEM),
                make_tensor_type((128, 4), DType.f32, storage=StorageKind.SMEM),
            )
        )


def test_verify_rejects_negative_wait() -> None:
    with pytest.raises(VerifyError, match="non-negative"):
        verify_prim_function(_wait_pf(-1))


# ── module for emit + GPU staging ────────────────────────────────────────────


@module(entry="async_stage_host")
class AsyncStage:
    @prim_func(target="cuda")
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

    @prim_func(target="cpu")
    def async_stage_host(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        launch(async_stage_device, a, b, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_async_copy_emits_cp_async() -> None:
    """The kernel forwards ``copy_async`` to the runtime entry and emits the
    group fences for commit / wait."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(AsyncStage, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source
    assert "tilefoundry::ops::copy_async(" in src
    assert "cp.async.commit_group;" in src
    assert "cp.async.wait_group %0;" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_copy_stage_matches_input() -> None:
    """A ``copy_async -> commit -> wait`` staging of a Split gmem source into a
    full shared tile reproduces the input (matches a synchronous copy)."""
    rm = tilefoundry.compile(AsyncStage, target="cuda")
    torch.manual_seed(0)
    a = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    rm(a, b)
    torch.cuda.synchronize()
    assert torch.allclose(b, a, rtol=0, atol=0)
