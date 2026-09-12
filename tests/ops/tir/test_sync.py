"""Exercise the CUDA barrier forms emitted for ``T.sync``.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.sync import SyncSquare
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="slice_host", target=_CUDA)
class SyncSlices:
    @prim_func(target=_CUDA)
    def one_warp(a: Tensor[(1, 32), "f32"], o: Tensor[(1, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1))) as full:
            with full[0:1] as m:
                sl = ShardLayout(Layout((1, 32), (32, 1)), (S(0), S(1)), m)
                src = T.tensor_view(a, layout=sl)
                dst = T.tensor_view(o, layout=sl)
                reg = T.alloc_tensor(Tensor[(1, 32), "f32", sl, "rmem"])
                T.copy(src, reg)
                T.sync(m)
                T.copy(reg, dst)

    @prim_func(target=_CUDA)
    def two_warps(a: Tensor[(2, 32), "f32"], o: Tensor[(2, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1))) as full:
            with full[2:4] as m:
                sl = ShardLayout(Layout((2, 32), (32, 1)), (S(0), S(1)), m)
                src = T.tensor_view(a, layout=sl)
                dst = T.tensor_view(o, layout=sl)
                reg = T.alloc_tensor(Tensor[(2, 32), "f32", sl, "rmem"])
                T.copy(src, reg)
                T.sync(m)
                T.copy(reg, dst)

    @prim_func(target=CpuTarget())
    def slice_host(
        a0: Tensor[(1, 32), "f32"],
        o0: Tensor[(1, 32), "f32"],
        a1: Tensor[(2, 32), "f32"],
        o1: Tensor[(2, 32), "f32"],
    ):
        launch(one_warp, a0, o0, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
        launch(two_warps, a1, o1, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


@module(entry="grid_sync_host", target=_CUDA)
class GridSync:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def grid_sync_device(a: Tensor[(128,), "f32"]):
        with Mesh((Topology("cta", 4),), Layout(shape=(4,), strides=(1,))) as m:
            T.sync(m)

    @prim_func(target=CpuTarget())
    def grid_sync_host(a: Tensor[(128,), "f32"]):
        launch(grid_sync_device, a, grid=(4, 1, 1), block=(128, 1, 1))  # noqa: F821


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sync_kernel_runs_and_squares() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x.square()
    runtime = tilefoundry.compile(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warp_slices_exchange_every_participants_value() -> None:
    one = torch.arange(32, dtype=torch.float32, device="cuda").reshape(1, 32)
    two = torch.arange(64, dtype=torch.float32, device="cuda").reshape(2, 32) + 1000
    one_out = torch.full((1, 32), -1.0, device="cuda")
    two_out = torch.full((2, 32), -1.0, device="cuda")
    runtime = tilefoundry.compile(SyncSlices, target=_CUDA)
    runtime(one, one_out, two, two_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(one_out, one, rtol=0, atol=0)
    torch.testing.assert_close(two_out, two, rtol=0, atol=0)
