from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm_host", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 32),)
)
class Fp8Gemm:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm_device(
        a: Tensor[(16, 128), "fp8e4m3"], b: Tensor[(8, 128), "fp8e4m3"], c: Tensor[(16, 8), "f32"]
    ):
        with Mesh((Topology("thread", 32),), layout=((4, 8), (1, 4)), names=("warp", "lane")) as m:
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32),
                    "fp8e4m3",
                    ((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 8), "fp8e4m3", ((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1)), "rmem"
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1)), "rmem"
                ]
            )
            T.fill(acc, 0.0)
            for k in range(0, 128, 32):
                a_slice = T.tensor_view(
                    T.ptr_of(a[0:0 + 16, k:k + 32]),
                    layout=((2, 8 @ m.lane, 4 @ m.warp, 8), (1024, 128, 8, 1)),
                    shape=(2, 8, 4, 8),
                )
                a_staged = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=(2, 8 @ m.lane, 4 @ m.warp, 8),
                    shape=(2, 8, 4, 8),
                )
                T.copy_async(a_slice, a_staged)
                T.cp_async_commit()
                T.cp_async_wait(n=0)
                T.sync(m)
                a_operand = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=(2, 8 @ m.lane, 2, 4 @ m.warp, 4),
                    shape=(16, 32),
                )
                T.copy(a_operand, a_frag)
                T.sync(m)
                b_slice = T.tensor_view(
                    T.ptr_of(b[0:0 + 8, k:k + 32]),
                    layout=((8 @ m.lane, 4 @ m.warp, 8), (128, 8, 1)),
                    shape=(8, 4, 8),
                )
                b_staged = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=(8 @ m.lane, 4 @ m.warp, 8),
                    shape=(8, 4, 8),
                )
                T.copy_async(b_slice, b_staged)
                T.cp_async_commit()
                T.cp_async_wait(n=0)
                T.sync(m)
                b_operand = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=((2, 4 @ m.warp, 4, 8 @ m.lane), (16, 4, 1, 32)),
                    shape=(32, 8),
                )
                T.copy(b_operand, b_frag)
                T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm89.Mma())
                T.sync(m)
            c_view = T.tensor_view(T.ptr_of(c), layout=(2, 8 @ m.lane, 4 @ m.warp, 2))
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def gemm_host(
        a: Tensor[(16, 128), "fp8e4m3"], b: Tensor[(8, 128), "fp8e4m3"], c: Tensor[(16, 8), "f32"]
    ):
        launch(gemm_device, a, b, c, grid=(
        1,
        1,
        1,
    ), block=(
        32,
        1,
        1,
    ), dynamic_smem=512)  # noqa: F821
