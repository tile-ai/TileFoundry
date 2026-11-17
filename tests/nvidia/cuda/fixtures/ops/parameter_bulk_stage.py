"""A (3, 2) parameter staged whole through the strided bulk tier, then read back."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir import types
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_CUDA = cuda.Target("nvidia.h200_sxm")


@module(
    entry="parameter_bulk_stage_host",
    target=_CUDA,
    topologies=(Topology("thread", 32),),
)
class ParameterBulkStage:
    """Stage a parameter into a shared tile of its shape; 24 bytes take the strided tier."""

    @prim_func(target=_CUDA)
    def parameter_bulk_stage_device(a: Tensor[(3, 2), "f32"], out: Tensor[(3, 2), "f32"]):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as m:
            stage = T.alloc_tensor(
                Tensor[
                    (3, 2),
                    "f32",
                    ShardLayout(
                        layout=Layout(shape=(3, 2), strides=(2, 1)),
                        attrs=(types.Broadcast(),),
                        mesh=m,
                    ),
                    "smem",
                ]
            )
            bar = T.alloc_tensor(Tensor[(1,), "i64", None, "smem"])
            T.mbarrier_init(bar, arrive_count=1)
            T.sync(m)
            T.copy_async_bulk(a, stage, bar)
            T.mbarrier_wait_parity(bar, 0)
            rows = T.tensor_view(T.ptr_of(out), layout=((3, 2), (2, 1)))
            T.copy(stage, rows)

    @prim_func(target=cpu.Target())
    def parameter_bulk_stage_host(a: Tensor[(3, 2), "f32"], out: Tensor[(3, 2), "f32"]):
        launch(parameter_bulk_stage_device, a, out, grid=(1, 1, 1), block=(32, 1, 1))
