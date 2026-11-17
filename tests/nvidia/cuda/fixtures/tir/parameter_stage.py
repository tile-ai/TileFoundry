"""A parameter staged whole into a shared tile of its own layout."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda


@module(
    entry="parameter_stage_host",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("thread", 1),),
)
class ParameterStage:
    @prim_func(target=cuda.Target("nvidia.h200_sxm"))
    def parameter_stage_device(a: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32"]):
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as _thread:
            shared = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", "smem"])
            T.copy_async(a, shared)
            T.cp_async_commit()
            T.cp_async_wait(n=0)
            out_view = T.tensor_view(T.ptr_of(out), layout=((2, 4), (4, 1)))
            T.copy(shared, out_view)

    @prim_func(target=cpu.Target())
    def parameter_stage_host(a: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32"]):
        launch(parameter_stage_device, a, out, grid=(1, 1, 1), block=(1, 1, 1))
