"""A parameter staged by warps that share a barrier, or that each hold a part of one."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

_CUDA = cuda.Target("nvidia.h200_sxm")


@module(entry="sliced_barrier_host", target=_CUDA, topologies=(Topology("thread", 96),))
class SlicedBarrierStage:
    """Two of three warps stage on a barrier marked Broadcast over their slice.

    One thread initialises the barrier and publishes it to the async proxy before the slice
    syncs. Both warps wait on it, and one thread writes the tile back.
    """

    @prim_func(target=_CUDA)
    def sliced_barrier_device(a: Tensor[(8,), "f32"], out: Tensor[(8,), "f32"]):
        with Mesh((Topology("thread", 96),), Layout((3, 32), (32, 1))) as full:
            with full[0:2] as pair:
                word = ShardLayout(Layout((1,), (1,)), (Broadcast(), Broadcast()), pair)
                stage = T.alloc_tensor(Tensor[(8,), "f32", None, "smem"])
                bar = T.alloc_tensor(Tensor[(1,), "i64", word, "smem"])
                with full[:1, :1] as _first:
                    T.mbarrier_init(bar, arrive_count=1)
                    T.fence_proxy_async()
                T.sync(pair)
                T.copy_async_bulk(a, stage, bar)
                T.mbarrier_wait_parity(bar, 0)
                with full[:1, :1] as _writer:
                    T.copy(stage, out)

    @prim_func(target=cpu.Target())
    def sliced_barrier_host(a: Tensor[(8,), "f32"], out: Tensor[(8,), "f32"]):
        launch(sliced_barrier_device, a, out, grid=(1, 1, 1), block=(96, 1, 1))


@module(entry="split_barrier_host", target=_CUDA, topologies=(Topology("thread", 64),))
class SplitBarrierStage:
    """The barrier is split over the warps, leaving the second warp a word no copy arrives on."""

    @prim_func(target=_CUDA)
    def split_barrier_device(a: Tensor[(8,), "f32"], out: Tensor[(8,), "f32"]):
        with Mesh((Topology("thread", 64),), Layout((2, 32), (32, 1))) as m:
            words = ShardLayout(Layout((2,), (1,)), (S(0), Broadcast()), m)
            stage = T.alloc_tensor(Tensor[(8,), "f32", None, "smem"])
            bar = T.alloc_tensor(Tensor[(2,), "i64", words, "smem"])
            T.mbarrier_init(bar, arrive_count=1)
            T.sync(m)
            T.copy_async_bulk(a, stage, bar)
            T.mbarrier_wait_parity(bar, 0)
            T.copy(stage, out)

    @prim_func(target=cpu.Target())
    def split_barrier_host(a: Tensor[(8,), "f32"], out: Tensor[(8,), "f32"]):
        launch(split_barrier_device, a, out, grid=(1, 1, 1), block=(64, 1, 1))
