"""``ops::mma`` on explicitly gathered per-lane fragments.

See [runtime §2.6](targets/cuda/tilefoundry_cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry_cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tests.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tests.fixtures.tir.mma import MmHandwritten
from tilefoundry.dsl import *
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget
from tilefoundry_cuda import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_MESH_LAYOUT = Layout(shape=(4, 8), strides=(1, 4))


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_handwritten_mma_matches_torch(target: CudaTarget) -> None:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime = tilefoundry.compile(retarget(MmHandwritten, target), target=target)
    runtime(a, b, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=2e-2, atol=2e-2)


@module(entry="tile_host", target=_CUDA, topologies=(Topology("thread", 32),))
class MmaTile:
    """A 16x16 by 16x8 product done as one atom."""

    @prim_func(target=_CUDA)
    def tile_device(
        a: Tensor[(256,), "bf16"],
        b: Tensor[(128,), "bf16"],
        c: Tensor[(128,), "f32"],
    ):
        with Mesh(
            (Topology("thread", 32),),
            _MESH_LAYOUT,
            names=("warp", "lane"),
        ) as m:
            a_view = T.tensor_view(
                T.ptr_of(a),
                shape=(16, 16),
                layout=ShardLayout(
                    layout=Layout(shape=(16, 16), strides=(16, 1)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=m,
                ),
            )
            b_view = T.tensor_view(
                T.ptr_of(b),
                shape=(16, 8),
                layout=ShardLayout(
                    layout=Layout(shape=(16, 8), strides=(8, 1)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=m,
                ),
            )
            a_tile = T.alloc_tensor(
                Tensor[
                    (16, 16),
                    "bf16",
                    ShardLayout(
                        layout=Layout(shape=(16, 16), strides=(16, 1)),
                        attrs=(Broadcast(), Broadcast()),
                        mesh=m,
                    ),
                    "smem",
                ]
            )
            b_tile = T.alloc_tensor(
                Tensor[
                    (16, 8),
                    "bf16",
                    ShardLayout(
                        layout=Layout(shape=(16, 8), strides=(8, 1)),
                        attrs=(Broadcast(), Broadcast()),
                        mesh=m,
                    ),
                    "smem",
                ]
            )
            a_frag = T.alloc_tensor(
                Tensor[
                    (16, 16),
                    "bf16",
                    ((2, 8 @ m.lane, 2, 4 @ m.warp, 2), (128, 16, 8, 2, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                Tensor[
                    (16, 8),
                    "bf16",
                    ((2, 4 @ m.warp, 2, 8 @ m.lane), (64, 16, 8, 1)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                Tensor[
                    (16, 8),
                    "f32",
                    ((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1)),
                    "rmem",
                ]
            )
            T.copy(a_view, a_tile)
            T.copy(b_view, b_tile)
            T.sync(m)
            b_fragment_view = T.tensor_view(
                T.ptr_of(b_tile),
                layout=((2, 4 @ m.warp, 2, 8 @ m.lane), (64, 16, 8, 1)),
            )
            T.ldmatrix(a_tile, a_frag)
            T.copy(b_fragment_view, b_frag)
            T.fill(acc, 0.0)
            T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm80.Mma())
            c_view = T.tensor_view(
                T.ptr_of(c), shape=(16, 8), layout=((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def tile_host(
        a: Tensor[(256,), "bf16"],
        b: Tensor[(128,), "bf16"],
        c: Tensor[(128,), "f32"],
    ):
        launch(tile_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_mma_matches_torch_matmul(target: CudaTarget) -> None:
    rm = tilefoundry.compile(retarget(MmaTile, target), target=target)
    torch.manual_seed(0)
    a = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(128, dtype=torch.float32, device="cuda")
    rm(a, b, c)
    torch.cuda.synchronize()
    expected = a.view(16, 16).float() @ b.view(16, 8).float()
    assert torch.allclose(c.view(16, 8), expected, rtol=2e-2, atol=2e-2)


@module(entry="fp8_host", target=_CUDA, topologies=(Topology("thread", 32),))
class Fp8MmaTile:
    """A 16x32 by 32x8 E4M3 product done as one atom, accumulating in F32."""

    @prim_func(target=_CUDA)
    def fp8_device(
        a: Tensor[(16, 32), "fp8e4m3"],
        b: Tensor[(32, 8), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        with Mesh((Topology("thread", 32),), _MESH_LAYOUT, names=("warp", "lane")) as m:
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32),
                    "fp8e4m3",
                    ((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 8),
                    "fp8e4m3",
                    ((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1)), "rmem"
                ]
            )
            a_view = T.tensor_view(
                T.ptr_of(a), layout=((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1))
            )
            b_view = T.tensor_view(
                T.ptr_of(b), layout=((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1))
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm89.Mma())
            c_view = T.tensor_view(
                T.ptr_of(c), layout=((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def fp8_host(
        a: Tensor[(16, 32), "fp8e4m3"],
        b: Tensor[(32, 8), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        launch(fp8_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_fragment_mma_matches_torch_matmul(target: CudaTarget) -> None:
    """E4M3 operands give the F32 product of the same values."""
    torch.manual_seed(1)
    a = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(32, 8, device="cuda").to(torch.float8_e4m3fn)
    c = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime = tilefoundry.compile(retarget(Fp8MmaTile, target), target=target)
    runtime(a, b, c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=2e-2, atol=2e-2)
