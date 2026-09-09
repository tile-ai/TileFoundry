"""Exercise every CUDA barrier form emitted for ``T.sync``.

A 128-thread mesh and three slices of it cover the whole block, a single warp,
a multi-warp run based at zero, and a multi-warp run based inside the block --
the four sets ``ops::sync`` picks a different barrier for. Each emits the mesh
and nothing else, so what the assertions read is the size and base the runtime
decides from. Successful completion plus correct output catches a barrier the
wrong threads arrive at, and a deadlock.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.tir.sync import SyncSquare
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="warp_slice_host")
class WarpSlice:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def warp_slice_device(a: Tensor[(128,), "f32"], o: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1))) as full:
            with full[2:4] as m:
                T.sync(m)

    @prim_func(target=CpuTarget())
    def warp_slice_host(a: Tensor[(128,), "f32"], o: Tensor[(128,), "f32"]):
        launch(warp_slice_device, a, o, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


@module(entry="grid_sync_host")
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
def test_warp_slice_named_barrier_runs() -> None:
    """Successful completion is the assertion; the wrong base deadlocks."""
    x = torch.arange(128, dtype=torch.float32, device="cuda")
    out = torch.full_like(x, -1)
    runtime = tilefoundry.compile(WarpSlice, target=CudaTarget("nvidia.h200_sxm"))
    runtime(x, out)
    torch.cuda.synchronize()
