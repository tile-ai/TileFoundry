"""Both staging tiers in one device function: a bulk run and a strided odd-sized run."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(
    entry="tma_tiers_host",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("thread", 128),),
)
class TmaTiers:
    """Both staging tiers in one device function, one operand pair each."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def tma_tiers_device(
        bulk_a: Tensor[(256,), "f32"],
        bulk_b: Tensor[(256,), "f32"],
        odd_a: Tensor[(5,), "f32"],
        odd_b: Tensor[(5,), "f32"],
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as m:
            bulk_view = T.tensor_view(
                T.ptr_of(bulk_a),
                layout=ShardLayout(
                    layout=Layout(shape=(256,), strides=(1,)), attrs=(Broadcast(),), mesh=m
                ),
            )
            bulk_stage = T.alloc_tensor(
                Tensor[
                    (256,),
                    "f32",
                    ShardLayout(
                        layout=Layout(shape=(256,), strides=(1,)),
                        attrs=(Broadcast(),),
                        mesh=m,
                    ),
                    "smem",
                ]
            )
            bulk_bar = T.alloc_tensor(Tensor[(1,), "i64", None, "smem"])
            T.mbarrier_init(bulk_bar, arrive_count=1)
            T.sync(m)
            T.copy_async_bulk(bulk_view, bulk_stage, bulk_bar)
            T.mbarrier_wait_parity(bulk_bar, 0)
            T.copy(bulk_stage, bulk_b)
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mo:
            odd_view = T.tensor_view(
                T.ptr_of(odd_a),
                layout=ShardLayout(
                    layout=Layout(shape=(5,), strides=(1,)), attrs=(Broadcast(),), mesh=mo
                ),
            )
            odd_stage = T.alloc_tensor(
                Tensor[
                    (5,),
                    "f32",
                    ShardLayout(
                        layout=Layout(shape=(5,), strides=(1,)),
                        attrs=(Broadcast(),),
                        mesh=mo,
                    ),
                    "smem",
                ]
            )
            odd_bar = T.alloc_tensor(Tensor[(1,), "i64", None, "smem"])
            T.mbarrier_init(odd_bar, arrive_count=1)
            T.sync(mo)
            T.copy_async_bulk(odd_view, odd_stage, odd_bar)
            T.mbarrier_wait_parity(odd_bar, 0)
            T.copy(odd_stage, odd_b)

    @prim_func(target=CpuTarget())
    def tma_tiers_host(
        bulk_a: Tensor[(256,), "f32"],
        bulk_b: Tensor[(256,), "f32"],
        odd_a: Tensor[(5,), "f32"],
        odd_b: Tensor[(5,), "f32"],
    ):
        launch(  # noqa: F821
            tma_tiers_device,  # noqa: F821
            bulk_a,
            bulk_b,
            odd_a,
            odd_b,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )
