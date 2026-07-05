"""GPU e2e for ``T.sync`` over a (4, 32) thread-128 mesh.

A hand-authored 128-thread ``@prim_func`` exercises every barrier form the sync
codegen produces, on real hardware:

- ``T.sync(m)``        → ``__syncthreads()`` (whole CTA),
- ``T.sync(m[0, :])``  → ``__syncwarp(mask)`` under a participant predicate,
- ``T.sync(m[0:2, :])``→ a named ``bar.sync`` (warps 0-1),
- ``T.sync(m[2:4, :])``→ a second named ``bar.sync`` (warps 2-3).

Each thread owns one element of a (4, 32) tensor (split over the warp×lane
mesh) and squares it. The square is barrier-independent by construction —
hand-written TIR has no cross-thread shared-memory surface yet — so the test's
value is that all four barrier forms compile to valid CUDA, allocate distinct
named-barrier ids, and run to completion (a mis-emitted ``bar.sync`` would
deadlock or fault) while the data result stays correct.
"""
from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology


@module(entry="sync_square_host")
class SyncSquare:
    @prim_func(target="cuda")
    def sync_square_device(a: Tensor[(4, 32), "f32"]):
        with Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")) as m:
            view = T.tensor_view(
                a,
                layout=ShardLayout(
                    layout=Layout(shape=(4, 32), strides=(32, 1)),
                    attrs=(Split(0), Split(1)),
                    mesh=m,
                ),
            )
            reg = T.alloc_tensor(
                TensorType(
                    shape=(4, 32),
                    dtype=DType.f32,
                    layout=ShardLayout(
                        layout=Layout(shape=(4, 32), strides=(32, 1)),
                        attrs=(Split(0), Split(1)),
                        mesh=m,
                    ),
                    storage=StorageKind.RMEM,
                )
            )
            T.copy(view, reg)
            T.sync(m)            # whole CTA  -> __syncthreads()
            T.sync(m[0, :])      # warp 0     -> __syncwarp(mask) + predicate
            T.sync(m[0:2, :])    # warps 0-1  -> bar.sync <id1>, 64
            T.sync(m[2:4, :])    # warps 2-3  -> bar.sync <id2>, 64
            T.binary(reg, reg, reg, kind=BinaryKind.MUL)
            T.copy(reg, view)

    @prim_func(target="cpu")
    def sync_square_host(a: Tensor[(4, 32), "f32"]):
        launch(sync_square_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_sync_barrier_forms_emit_expected_cuda() -> None:
    """The kernel lowers each barrier form to the expected CUDA, with two
    distinct named-barrier ids for the two multi-warp groups."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(SyncSquare, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source

    assert "__syncthreads();" in src
    assert "__syncwarp(0xffffffffu)" in src
    assert "bar.sync" in src
    assert '"r"(1)' in src and '"r"(2)' in src  # two distinct named ids
    assert '"r"(64)' in src                      # 64-thread participant count


@module(entry="grid_sync_host")
class GridSync:
    @prim_func(target="cuda")
    def grid_sync_device(a: Tensor[(128,), "f32"]):
        with Mesh(Topology("cta", 4), Layout(shape=(4,), strides=(1,))) as m:
            T.sync(m)

    @prim_func(target="cpu")
    def grid_sync_host(a: Tensor[(128,), "f32"]):
        launch(grid_sync_device, a, grid=(4, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_grid_scope_sync_emits_grid_barrier() -> None:
    """A ``T.sync`` over a full ``cta``-topology mesh lowers to the grid-wide
    software barrier helper (not a within-block ``__syncthreads``), and the
    module defines its own internal-linkage counter for it."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(GridSync, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source
    assert "tilefoundry::ops::grid_barrier(tilefoundry::tf_grid_bar_state);" in src
    # The backing counter is defined per module with internal linkage, not
    # pulled from a shared header global.
    assert "static __device__ unsigned int tf_grid_bar_state[2];" in src


def test_sync_kernel_runs_and_squares() -> None:
    """All four barrier forms compile and run on GPU without deadlock/fault,
    and the elementwise square is correct."""
    rm = tilefoundry.compile(SyncSquare, target="cuda")
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x * x
    rm(x)
    torch.cuda.synchronize()
    assert torch.allclose(x, expected, rtol=0, atol=0)
