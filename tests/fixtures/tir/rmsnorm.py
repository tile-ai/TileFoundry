from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import B, Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="rmsnorm_host", target=CudaTarget("nvidia.h200_sxm"))
class TirRmsnorm:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def rmsnorm_device(x: Tensor[(1, 128), "f32"], weight: Tensor[(128,), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,)), names=('t',)) as thread:
            x_view = T.tensor_view(x, layout=((1, 128), (128, 1), {thread.t @ B()}))
            weight_view = T.tensor_view(weight, layout=((128,), (1,), {thread.t @ B()}))
            out_view = T.tensor_view(out, layout=((1, 128), (128, 1), {thread.t @ B()}))
            T.rms_norm(x_view, out_view, weight_view, eps=1e-05)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def rmsnorm_host(x: Tensor[(1, 128), "f32"], weight: Tensor[(128,), "f32"], out: Tensor[(1, 128), "f32"]):
        launch(rmsnorm_device, x, weight, out, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821
