from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import B, Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="async_stage_host")
class AsyncStage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def async_stage_device(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,)), names=('t',)) as m:
            a_view = T.tensor_view(a, layout=ShardLayout(layout=Layout(shape=(128, 4), strides=(4, 1)), attrs=(S(0),), mesh=Mesh(topologies=(Topology(name="thread", size=128),), layout=Layout(shape=(128,), strides=(1,)), names=("t",))))
            shared = T.alloc_tensor(tensor_type=Tensor[(512,), "f32",
    ShardLayout(
        layout=Layout((512,), (1,)),
        attrs=(B(),),
        mesh=Mesh((Topology("thread", 128),), Layout((128,), (1,)), names=('t',)),
    ), "smem"])
            T.copy_async(a_view, shared)
            T.cp_async_commit()
            T.cp_async_wait(n=0)
            T.sync(m)
            T.copy(shared, b)

    @prim_func(target=CpuTarget())
    def async_stage_host(a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]):
        launch(async_stage_device, a, b, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
