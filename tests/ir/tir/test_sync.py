"""Exercise every CUDA barrier form emitted for ``T.sync``.

A 128-thread mesh covers whole CTA, single-warp subset, and two named multi-warp
barriers. Successful completion plus correct output catches invalid masks,
barrier ids, and deadlocks.

See [tir §1.5](docs/spec/tir.md#15-sync).
"""

from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="sync_square_host")
class SyncSquare:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def sync_square_device(a: Tensor[(4, 32), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as m:
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
            T.sync(m)
            T.sync(m[0, :])
            T.sync(m[0:2, :])
            T.sync(m[2:4, :])
            T.binary(reg, reg, reg, kind=BinaryKind.MUL)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def sync_square_host(a: Tensor[(4, 32), "f32"]):
        launch(sync_square_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_sync_barrier_forms_emit_expected_cuda() -> None:
    """The kernel lowers each barrier form to the expected CUDA.

    The kernel lowers each barrier form to the expected CUDA, with two
    distinct named-barrier ids for the two multi-warp groups.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source

    assert "SyncKind::syncthreads>();" in src
    assert "SyncKind::syncwarp_masked, 0, 32, 0xffffffffu>();" in src

    assert "SyncKind::bar_sync, 0, 64, 0u, 1>();" in src
    assert "SyncKind::bar_sync, 64, 64, 0u, 2>();" in src


@module(entry="grid_sync_host")
class GridSync:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def grid_sync_device(a: Tensor[(128,), "f32"]):
        with Mesh((Topology("cta", 4),), Layout(shape=(4,), strides=(1,))) as m:
            T.sync(m)

    @prim_func(target=CpuTarget())
    def grid_sync_host(a: Tensor[(128,), "f32"]):
        launch(grid_sync_device, a, grid=(4, 1, 1), block=(128, 1, 1))  # noqa: F821


def test_grid_scope_sync_emits_grid_barrier() -> None:
    """Test grid scope sync emits grid barrier.

    A ``T.sync`` over a full ``cta``-topology mesh lowers to the grid-wide
    software barrier helper (not a within-block ``__syncthreads``), and the
    module defines its own internal-linkage counter for it.
    """
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(GridSync, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source
    assert (
        "tilefoundry::ops::sync<tilefoundry::ops::SyncKind::grid>"
        "(tilefoundry::tf_grid_bar_state);" in src
    )

    assert "static __device__ unsigned int tf_grid_bar_state[2];" in src


def test_sync_kernel_runs_and_squares() -> None:
    """All four barrier forms compile and run on GPU without deadlock/fault.

    All four barrier forms compile and run on GPU without deadlock/fault,
    and the elementwise square is correct.
    """
    rm = tilefoundry.compile(SyncSquare, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    expected = x * x
    rm(x)
    torch.cuda.synchronize()
    assert torch.allclose(x, expected, rtol=0, atol=0)
