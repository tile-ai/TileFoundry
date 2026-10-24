from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(
    entry="async_stage_host",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("thread", 128),),
)
class AsyncStage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def async_stage_device(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        with Mesh((Topology("thread", 128),), layout=(128,), names=("t",)) as m:
            a_view = T.tensor_view(T.ptr_of(a), layout=(128 @ m.t, 4))
            shared = T.alloc_tensor(tensor_type=Tensor[(128, 4), "f32", "smem"])
            shared_view = T.tensor_view(T.ptr_of(shared), layout=(128 @ m.t, 4))
            b_view = T.tensor_view(T.ptr_of(b), layout=(128 @ m.t, 4))
            T.copy_async(a_view, shared_view)
            T.cp_async_commit()
            T.cp_async_wait(n=0)
            T.sync(m)
            T.copy(shared_view, b_view)

    @prim_func(target=CpuTarget())
    def async_stage_host(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        launch(async_stage_device, a, b, grid=(
        1,
        1,
        1,
    ), block=(
        128,
        1,
        1,
    ))  # noqa: F821
