"""``ops::mma`` on explicitly gathered per-lane fragments.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.mma import MmHandwritten
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_MESH_LAYOUT = Layout(shape=(4, 8), strides=(1, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_handwritten_mma_matches_torch() -> None:
    torch.manual_seed(2)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime = tilefoundry.compile(MmHandwritten, target=CudaTarget("nvidia.h200_sxm"))
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
                layout=ShardLayout(
                    layout=Layout(shape=(256,), strides=(1,)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=m,
                ),
            )
            b_view = T.tensor_view(
                T.ptr_of(b),
                layout=ShardLayout(
                    layout=Layout(shape=(128,), strides=(1,)),
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
                    ((2, 4 @ m.warp, 2, 8 @ m.lane, 2), (1, 2, 8, 16, 128)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                Tensor[
                    (16, 8),
                    "bf16",
                    ((8 @ m.lane, 2, 4 @ m.warp, 2), (1, 8, 16, 64)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                Tensor[
                    (16, 8),
                    "f32",
                    ((2, 4 @ m.warp, 8 @ m.lane, 2), (1, 2, 8, 64)),
                    "rmem",
                ]
            )
            T.copy(a_view, a_tile)
            T.copy(b_view, b_tile)
            T.sync(m)
            b_fragment_view = T.tensor_view(
                T.ptr_of(b_tile),
                layout=((8 @ m.lane, 2, 4 @ m.warp, 2), (1, 8, 16, 64)),
            )
            T.ldmatrix(a_tile, a_frag)
            T.copy(b_fragment_view, b_frag)
            T.fill(acc, 0.0)
            T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm80.Mma())
            c_view = T.tensor_view(
                T.ptr_of(c), layout=((2, 4 @ m.warp, 8 @ m.lane, 2), (1, 2, 8, 64))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def tile_host(
        a: Tensor[(256,), "bf16"],
        b: Tensor[(128,), "bf16"],
        c: Tensor[(128,), "f32"],
    ):
        launch(tile_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_mma_matches_torch_matmul() -> None:
    rm = tilefoundry.compile(MmaTile, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(128, dtype=torch.float32, device="cuda")
    rm(a, b, c)
    torch.cuda.synchronize()
    expected = a.view(16, 16).float() @ b.view(16, 8).float()
    assert torch.allclose(c.view(16, 8), expected, rtol=2e-2, atol=2e-2)
