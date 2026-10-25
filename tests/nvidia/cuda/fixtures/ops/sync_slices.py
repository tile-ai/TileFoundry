"""Two warp slices of one 128-thread mesh, each copying through a synced register tile."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="slice_host", target=_CUDA, topologies=(Topology("thread", 128),))
class SyncSlices:
    @prim_func(target=_CUDA)
    def one_warp(a: Tensor[(1, 32), "f32"], o: Tensor[(1, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1))) as full:
            with full[0:1] as m:
                sl = ShardLayout(Layout((1, 32), (32, 1)), (S(0), S(1)), m)
                src = T.tensor_view(T.ptr_of(a), layout=sl)
                dst = T.tensor_view(T.ptr_of(o), layout=sl)
                reg = T.alloc_tensor(Tensor[(1, 32), "f32", sl, "rmem"])
                T.copy(src, reg)
                T.sync(m)
                T.copy(reg, dst)

    @prim_func(target=_CUDA)
    def two_warps(a: Tensor[(2, 32), "f32"], o: Tensor[(2, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1))) as full:
            with full[2:4] as m:
                sl = ShardLayout(Layout((2, 32), (32, 1)), (S(0), S(1)), m)
                src = T.tensor_view(T.ptr_of(a), layout=sl)
                dst = T.tensor_view(T.ptr_of(o), layout=sl)
                reg = T.alloc_tensor(Tensor[(2, 32), "f32", sl, "rmem"])
                T.copy(src, reg)
                T.sync(m)
                T.copy(reg, dst)

    @prim_func(target=CpuTarget())
    def slice_host(
        a0: Tensor[(1, 32), "f32"],
        o0: Tensor[(1, 32), "f32"],
        a1: Tensor[(2, 32), "f32"],
        o1: Tensor[(2, 32), "f32"],
    ):
        launch(one_warp, a0, o0, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
        launch(two_warps, a1, o1, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
