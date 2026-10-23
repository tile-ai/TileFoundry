from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget, CudaTarget


@module(
    entry="split_stage_host",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("thread", 128),),
)
class SplitStage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def split_stage_device(a: Tensor[(2, 128, 4), "f32"], b: Tensor[(2, 128, 4), "f32"]):
        with Mesh((Topology("thread", 128),), layout=(128,), names=("t",)) as m:
            a_view = T.tensor_view(T.ptr_of(a), layout=(2, 128 @ m.t, 4))
            shared = T.alloc_tensor(tensor_type=Tensor[(2, 128, 4), "f32", "smem"])
            shared_view = T.tensor_view(T.ptr_of(shared), layout=(2, 128 @ m.t, 4))
            b_view = T.tensor_view(T.ptr_of(b), layout=(2, 128 @ m.t, 4))
            T.copy_async(a_view, shared_view)
            T.cp_async_commit()
            T.cp_async_wait(n=0)
            T.sync(m)
            T.copy(shared_view, b_view)

    @prim_func(target=CpuTarget())
    def split_stage_host(a: Tensor[(2, 128, 4), "f32"], b: Tensor[(2, 128, 4), "f32"]):
        launch(split_stage_device, a, b, grid=(
        1,
        1,
        1,
    ), block=(
        128,
        1,
        1,
    ))  # noqa: F821
