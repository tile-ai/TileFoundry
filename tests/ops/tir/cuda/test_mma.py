"""``ops::mma``'s tile tier: the atom looped over a whole shared-memory tile.

The entry reads the tier off what a thread *holds*: rank-2 static A and B local
views are a tile, the atom's ``(8, 4, 4)`` lane fragments are the single
instruction. The Atom tier is already gated on GPU by
``tests/integration/test_mma_tir_handwritten.py``; forcing this one takes a
rank-2 view of the whole ``(M, K)`` and ``(N, K)``, so A and B are shared tiles
every thread broadcasts.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.mma import MmHandwritten
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_OP = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN

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


@module(entry="tile_host")
class MmaTile:
    """A 16x16 by 16x8 product done as one atom, looped by the tile's shape.

    The mesh is the atom's own: one warp as the ``(4, 8)`` the m16n8k16
    fragment map is written against, so 32 instances is one warp and all of N
    stays with it. The two tiles are fully broadcast, so each thread's local
    view is the allocation's rank-2 layout and ``tile_shaped_v`` is true.
    ``N`` comes off B's first mode, so B is ``(N, K)`` -- ``(8, 16)`` -- in its
    shape and its shard layout alike.
    """

    @prim_func(target=_CUDA)
    def tile_device(
        a: Tensor[(256,), "bf16"],
        b: Tensor[(128,), "bf16"],
        c: Tensor[(128,), "f32"],
    ):
        atom = T.cuda.mma.atom(op=_OP)
        with Mesh((Topology("thread", 32),), _MESH_LAYOUT) as m:
            a_view = T.tensor_view(
                a,
                layout=ShardLayout(
                    layout=Layout(shape=(256,), strides=(1,)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=m,
                ),
            )
            b_view = T.tensor_view(
                b,
                layout=ShardLayout(
                    layout=Layout(shape=(128,), strides=(1,)),
                    attrs=(Broadcast(), Broadcast()),
                    mesh=m,
                ),
            )
            a_tile = T.alloc_tensor(
                Tensor[(16, 16), 'bf16', ShardLayout(
                        layout=Layout(shape=(16, 16), strides=(16, 1)),
                        attrs=(Broadcast(), Broadcast()),
                        mesh=m,
                    ), 'smem']
            )
            b_tile = T.alloc_tensor(
                Tensor[(8, 16), 'bf16', ShardLayout(
                        layout=Layout(shape=(8, 16), strides=(1, 8)),
                        attrs=(Broadcast(), Broadcast()),
                        mesh=m,
                    ), 'smem']
            )
            acc = T.alloc_tensor(
                Tensor[(16, 8), 'f32', atom.C, 'rmem']
            )
            T.copy(a_view, a_tile)
            T.copy(b_view, b_tile)
            T.fill(acc, 0.0)
            T.sync(m)
            T.mma(acc, a_tile, b_tile)
            c_view = T.tensor_view(c, layout=atom.C)
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def tile_host(
        a: Tensor[(256,), "bf16"],
        b: Tensor[(128,), "bf16"],
        c: Tensor[(128,), "f32"],
    ):
        launch(tile_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tile_tier_matches_torch_matmul() -> None:
    """Selected by rank-2 static A and B local views over the whole tile.

    The two tiles are filled by a linear ``copy`` into a column-major shared
    allocation, so ``av(m, k)`` is ``a[m + 16k]`` and ``bv(n, k)`` is
    ``b[n + 8k]``: A is the flat source read column-major, which is
    ``a.view(16, 16).T``, and B's ``(N, K)`` view of ``b`` is the transpose of
    the ``(K, N)`` matrix ``b.view(16, 8)``. The product the tier computes is
    therefore ``a.view(16, 16).T @ b.view(16, 8)``.
    """
    rm = tilefoundry.compile(MmaTile, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(128, dtype=torch.float32, device="cuda")
    rm(a, b, c)
    torch.cuda.synchronize()
    expected = a.view(16, 16).float().t() @ b.view(16, 8).float()
    assert torch.allclose(c.view(16, 8), expected, rtol=2e-2, atol=2e-2)
