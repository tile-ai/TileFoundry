"""A tile staged through more dynamic shared memory than CUDA grants without opting in."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


def _rows(mesh) -> ShardLayout:
    """One contiguous 160-element row per thread."""
    return ShardLayout(Layout((128, 160), (160, 1)), (S(0),), mesh)


@module(entry="main", target=_CUDA, topologies=(Topology("thread", 128),))
class WideSharedStage:
    """One block stages a (128, 160) f32 tile, 80 KiB, through shared memory and back.

    The host entry requests the 80 KiB its stage uses.
    """

    @prim_func(target=_CUDA)
    def wide_stage(x: Tensor[(128, 160), "f32"], out: Tensor[(128, 160), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            source = T.tensor_view(T.ptr_of(x), layout=_rows(thread))
            stage = T.tensor_view(
                0, dtype="f32", storage="smem", layout=_rows(thread), shape=(128, 160)
            )
            target = T.tensor_view(T.ptr_of(out), layout=_rows(thread))
            T.copy(source, stage)
            T.sync(thread)
            T.copy(stage, target)

    @prim_func(target=CpuTarget())
    def main(x: Tensor[(128, 160), "f32"], out: Tensor[(128, 160), "f32"]):
        launch(wide_stage, x, out, grid=(1, 1, 1), block=(128, 1, 1), dynamic_smem=81920)
