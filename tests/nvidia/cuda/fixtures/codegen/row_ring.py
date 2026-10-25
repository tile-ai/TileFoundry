"""``staged_nested``'s six rows staged through a two-stage smem ring."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    entry="row_ring_host",
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class RowRing:
    """``staged_nested``'s six rows through the same two-stage ring.

    Kept as authored: row ``3i + j`` stages through slot ``(3i + j) % 2``,
    the row windows are slice-based views with a stride-0 first mode, and
    the ring index is dim arithmetic over the two loop counters.
    """

    @prim_func(target=_CUDA)
    def row_ring_device(x: Tensor[(6, 8), "bf16"], out: Tensor[(6, 8), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:  # noqa: F841
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                ring_stages = (
                    T.tensor_view(0, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                    T.tensor_view(16, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                )
                for i in range(0, 2, 1):
                    for j in range(0, 3, 1):
                        value = T.tensor_view(
                            T.ptr_of(x[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                            layout=((1, 8), (0, 1)),
                            shape=(1, 8),
                        )
                        stage = T.tensor_view(
                            T.ptr_of(ring_stages[((i * 3) + j) % 2]),
                            layout=(1, 8),
                            shape=(1, 8),
                        )
                        T.copy_async(value, stage)
                        T.cp_async_commit()
                        T.cp_async_wait(n=0)
                        T.sync(scope)
                        out_row = T.tensor_view(
                            T.ptr_of(out[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                            layout=((1, 8), (0, 1)),
                            shape=(1, 8),
                        )
                        T.copy(stage, out_row)

    @prim_func(target=CpuTarget())
    def row_ring_host(x: Tensor[(6, 8), "bf16"], out: Tensor[(6, 8), "bf16"]):
        launch(  # noqa: F821
            row_ring_device,  # noqa: F821
            x,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            dynamic_smem=32,
        )
