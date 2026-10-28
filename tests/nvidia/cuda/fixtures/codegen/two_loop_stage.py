"""``loop_outputs``'s two staging loops, run by one thread with the wait each read back needs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    entry="main",
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 1)),
)
class TwoLoopStage:
    """Each loop stages rows of ``x`` through its own view of shared-memory offset 0.

    One thread issues every copy and reads it back, so each byte has one writer. The
    second loop names the dynamic shared-memory base outside the block of the first.
    """

    @prim_func(target=_CUDA)
    def two_loops(
        x: Tensor[(2, 4), "bf16"], first: Tensor[(2, 4), "bf16"], second: Tensor[(2, 4), "bf16"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as _cta:
            with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as _thread:
                for row in range(0, 2, 1):
                    source = T.tensor_view(
                        T.ptr_of(x[row:row + 1, 0:0 + 1]), layout=(1, 4), shape=(1, 4)
                    )
                    landing = T.tensor_view(
                        0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                    )
                    T.copy_async(source, landing)
                    T.cp_async_commit()
                    T.cp_async_wait(n=0)
                    window = T.tensor_view(
                        T.ptr_of(first[row:row + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy(landing, window)
                for row_2 in range(0, 2, 1):
                    source_2 = T.tensor_view(
                        T.ptr_of(x[row_2:row_2 + 1, 0:0 + 1]), layout=(1, 4), shape=(1, 4)
                    )
                    landing_2 = T.tensor_view(
                        0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                    )
                    T.copy_async(source_2, landing_2)
                    T.cp_async_commit()
                    T.cp_async_wait(n=0)
                    window_2 = T.tensor_view(
                        T.ptr_of(second[row_2:row_2 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy(landing_2, window_2)

    @prim_func(target=CpuTarget())
    def main(x: Tensor[(2, 4), "bf16"], first: Tensor[(2, 4), "bf16"], second: Tensor[(2, 4), "bf16"]):
        launch(two_loops, x, first, second, grid=(1, 1, 1), block=(1, 1, 1), dynamic_smem=8)
