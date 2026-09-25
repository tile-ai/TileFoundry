from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="mm_host", target=CudaTarget("nvidia.h200_sxm"))
class MmHandwritten:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mm_device(
        a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]
    ):
        with Mesh(
            (Topology("thread", 32),), Layout((4, 8), (1, 4)), names=("warp", "lane")
        ) as _warp:
            a_view = T.tensor_view(
                T.ptr_of(a), layout=((2, 4 @ _warp.warp, 2, 8 @ _warp.lane, 2), (1, 2, 8, 16, 128))
            )
            b_view = T.tensor_view(
                T.ptr_of(b), layout=((8 @ _warp.lane, 2, 4 @ _warp.warp, 2), (1, 8, 16, 64))
            )
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "bf16",
                    ((2, 4 @ _warp.warp, 2, 8 @ _warp.lane, 2), (1, 2, 8, 16, 128)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8),
                    "bf16",
                    ((8 @ _warp.lane, 2, 4 @ _warp.warp, 2), (1, 8, 16, 64)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 4 @ _warp.warp, 8 @ _warp.lane, 2), (1, 2, 8, 64)), "rmem"
                ]
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.mma(
                acc,
                a_frag,
                b_frag,
                atom=T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN),
            )
            c_view = T.tensor_view(
                T.ptr_of(c), layout=((2, 4 @ _warp.warp, 8 @ _warp.lane, 2), (1, 2, 8, 64))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def mm_host(a: Tensor[(16, 16), "bf16"], b: Tensor[(16, 8), "bf16"], c: Tensor[(16, 8), "f32"]):
        launch(mm_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821
