"""A single-row TIR RMSNorm program with a deterministic torch oracle."""

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import B, Layout, Mesh, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="rmsnorm_host")
class TirRmsnorm:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def rmsnorm_device(
        x: Tensor[(1, 128), "f32"],
        weight: Tensor[(128,), "f32"],
        out: Tensor[(1, 128), "f32"],
    ):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,))) as thread:
            matrix = ShardLayout(Layout((1, 128), (128, 1)), (B(),), thread)
            vector = ShardLayout(Layout((128,), (1,)), (B(),), thread)
            x_view = T.tensor_view(x, layout=matrix)
            weight_view = T.tensor_view(weight, layout=vector)
            out_view = T.tensor_view(out, layout=matrix)
            T.rms_norm(x_view, out_view, weight_view, eps=1e-5)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def rmsnorm_host(
        x: Tensor[(1, 128), "f32"],
        weight: Tensor[(128,), "f32"],
        out: Tensor[(1, 128), "f32"],
    ):
        launch(
            rmsnorm_device,
            x,
            weight,
            out,
            grid=(1, 1, 1),
            block=(1, 1, 1),
        )  # noqa: F821
