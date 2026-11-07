"""A WGMMA unit followed by an ordinary unit in the same CUDA project."""

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(topologies=(Topology("thread", 128),))
class _Group:
    @prim_func(target=_CUDA)
    def device(x: Tensor[(128,), "f32"], out: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            T.wgmma_fence()
            T.wgmma_commit_group()
            T.wgmma_wait_group(n=0)
            layout = ShardLayout(Layout((128,), (1,)), (Split(0),), thread)
            src = T.tensor_view(T.ptr_of(x), layout=layout)
            dst = T.tensor_view(T.ptr_of(out), layout=layout)
            T.copy(src, dst)


@module(topologies=(Topology("thread", 32),))
class _Plain:
    @prim_func(target=_CUDA)
    def device(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
        with Mesh((Topology("thread", 32),), Layout((32,), (1,))) as thread:
            T.fence_proxy_async()
            T.fence_proxy_async(storage="gmem")
            layout = ShardLayout(Layout((32,), (1,)), (Split(0),), thread)
            src = T.tensor_view(T.ptr_of(x), layout=layout)
            dst = T.tensor_view(T.ptr_of(out), layout=layout)
            T.copy(src, dst)


@module(entry="host", target=_CUDA)
class MixedUnits:
    group = _Group
    plain = _Plain

    @prim_func(target=CpuTarget())
    def host(
        x0: Tensor[(128,), "f32"],
        o0: Tensor[(128,), "f32"],
        x1: Tensor[(32,), "f32"],
        o1: Tensor[(32,), "f32"],
    ):
        launch(group.device, x0, o0, grid=(1, 1, 1), block=(128, 1, 1))
        launch(plain.device, x1, o1, grid=(1, 1, 1), block=(32, 1, 1))
