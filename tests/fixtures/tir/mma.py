from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="mm_host")
class MmHandwritten:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mm_device(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        with Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))) as _warp:
            a_view = T.tensor_view(a, layout=ShardLayout(layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)), attrs=(S(1), S(3)), mesh=Mesh(topologies=(Topology(name="thread", size=32),), layout=Layout(shape=(4, 8), strides=(1, 4)), names=())))
            b_view = T.tensor_view(b, layout=ShardLayout(layout=Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64)), attrs=(S(2), S(0)), mesh=Mesh(topologies=(Topology(name="thread", size=32),), layout=Layout(shape=(4, 8), strides=(1, 4)), names=())))
            a_frag = T.alloc_tensor(tensor_type=Tensor[(16, 16), "bf16",
    ShardLayout(
        layout=Layout((2, 4, 2, 8, 2), (1, 2, 8, 16, 128)),
        attrs=(S(1), S(3)),
        mesh=Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))),
    ), "rmem"])
            b_frag = T.alloc_tensor(tensor_type=Tensor[(16, 8), "bf16",
    ShardLayout(
        layout=Layout((8, 2, 4, 2), (1, 8, 16, 64)),
        attrs=(S(2), S(0)),
        mesh=Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))),
    ), "rmem"])
            acc = T.alloc_tensor(tensor_type=Tensor[(16, 8), "f32",
    ShardLayout(
        layout=Layout((2, 4, 8, 2), (1, 2, 8, 64)),
        attrs=(S(1), S(2)),
        mesh=Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))),
    ), "rmem"])
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.mma(acc, a_frag, b_frag, atom=T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN))
            c_view = T.tensor_view(c, layout=ShardLayout(layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)), attrs=(S(1), S(2)), mesh=Mesh(topologies=(Topology(name="thread", size=32),), layout=Layout(shape=(4, 8), strides=(1, 4)), names=())))
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def mm_host(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        launch(mm_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821
