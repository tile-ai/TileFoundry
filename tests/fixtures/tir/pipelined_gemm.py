from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import B, Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="pipelined_host")
class PipelinedGemm:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def pipelined_device(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        with Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))) as mesh:
            a_global = T.tensor_view(a, layout=ShardLayout(layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)), attrs=(B(), B()), mesh=mesh))
            a_stage = T.alloc_tensor(tensor_type=Tensor[(16, 16), "bf16",
                ShardLayout(
                    layout=Layout((2, 4, 2, 8, 2), (1, 2, 8, 16, 128)),
                    attrs=(B(), B()),
                    mesh=mesh,
                ), "smem"])
            barrier = T.alloc_tensor(tensor_type=Tensor[(1,), "i64", "smem"])
            T.mbarrier_init(barrier, arrive_count=1)
            T.sync(mesh)
            T.tma_copy(a_global, a_stage, barrier)
            T.mbarrier_wait_parity(barrier, 0)
            b_view = T.tensor_view(b, layout=ShardLayout(layout=Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64)), attrs=(S(2), S(0)), mesh=mesh))
            a_view = T.tensor_view(a_stage, layout=ShardLayout(layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)), attrs=(S(1), S(3)), mesh=mesh))
            a_frag = T.alloc_tensor(tensor_type=Tensor[(16, 16), "bf16",
                ShardLayout(
                    layout=Layout((2, 4, 2, 8, 2), (1, 2, 8, 16, 128)),
                    attrs=(S(1), S(3)),
                    mesh=mesh,
                ), "rmem"])
            b_frag = T.alloc_tensor(tensor_type=Tensor[(16, 8), "bf16",
                ShardLayout(
                    layout=Layout((8, 2, 4, 2), (1, 8, 16, 64)),
                    attrs=(S(2), S(0)),
                    mesh=mesh,
                ), "rmem"])
            acc = T.alloc_tensor(tensor_type=Tensor[(16, 8), "f32",
                ShardLayout(
                    layout=Layout((2, 4, 8, 2), (1, 2, 8, 64)),
                    attrs=(S(1), S(2)),
                    mesh=mesh,
                ), "rmem"])
            c_view = T.tensor_view(c, layout=ShardLayout(layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)), attrs=(S(1), S(2)), mesh=mesh))
            T.copy(b_view, b_frag)
            T.copy(a_view, a_frag)
            T.fill(acc, 0.0)
            T.sync(mesh)
            T.mma(acc, a_frag, b_frag)
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def pipelined_host(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        launch(pipelined_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821
