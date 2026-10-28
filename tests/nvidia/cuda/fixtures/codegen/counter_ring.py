"""``staged_dynamic_inner``'s two-stage smem ring, driven by an rmem stage counter."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    entry="counter_ring_host",
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class CounterRing:
    """``staged_dynamic_inner``'s ring, driven by its rmem stage counter.

    Each stage view and the tuple have a binding. The counter modulo two
    selects a bound stage in the consuming loop scope.
    """

    @prim_func(target=_CUDA)
    def counter_ring_device(x: Tensor[(1, 8), "bf16"], out: Tensor[(1, 8), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:  # noqa: F841
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                ring_0 = T.tensor_view(0, dtype='bf16', storage='smem', layout=(1, 8), shape=(1, 8))
                ring_1 = T.tensor_view(16, dtype='bf16', storage='smem', layout=(1, 8), shape=(1, 8))
                ring_stages = (ring_0, ring_1)
                stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(stage_counter, 0)
                stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(stage_one, 1)
                for i in range(0, 2, 1):
                    value = T.tensor_view(T.ptr_of(x), layout=(1, 8), shape=(1, 8))
                    stage = ring_stages[stage_counter % 2]
                    T.copy_async(value, stage)
                    T.cp_async_commit()
                    T.cp_async_wait(n=0)
                    T.sync(scope)
                    out_view = T.tensor_view(T.ptr_of(out), layout=(1, 8), shape=(1, 8))
                    T.copy(stage, out_view)
                    T.binary(stage_counter, stage_one, stage_counter, kind=BinaryKind.ADD)

    @prim_func(target=CpuTarget())
    def counter_ring_host(x: Tensor[(1, 8), "bf16"], out: Tensor[(1, 8), "bf16"]):
        launch(  # noqa: F821
            counter_ring_device,  # noqa: F821
            x,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            dynamic_smem=32,
        )
