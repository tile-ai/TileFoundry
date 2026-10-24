from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 128), "bf16"], out: Tensor[(64, 128), "f32"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh(
            (Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
        ) as scope:
            T.fill(out, 0.0)
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 128),
                    "f32",
                    ((4 @ scope.d1, 2, 8 @ scope.d2, 2 @ scope.d0, 8, 4 @ scope.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                    "rmem",
                ]
            )
            T.fill(acc, 0.0)
            value_1 = T.tensor_view(T.ptr_of(a), layout=(64, 16), shape=(64, 16))
            value = T.tensor_view(
                4096,
                dtype="bf16",
                storage="smem",
                layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                shape=(64, 16),
            )
            value_2 = T.tensor_view(
                T.ptr_of(value),
                layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                shape=(64, 16),
            )
            T.copy(value_1, value_2)
            value_4 = T.tensor_view(T.ptr_of(b), layout=(16, 128), shape=(16, 128))
            value_3 = T.tensor_view(
                0,
                dtype="bf16",
                storage="smem",
                layout=Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 0, 1))) | Swizzle(3, 4, 3),
                shape=(16, 128),
            )
            value_5 = T.tensor_view(
                T.ptr_of(value_3),
                layout=Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 0, 1))) | Swizzle(3, 4, 3),
                shape=(16, 128),
            )
            T.copy(value_4, value_5)
            for o_m in range(0, 64, 64):
                for o_n in range(0, 128, 128):
                    for o_k in range(0, 16, 16):
                        value_10 = T.tensor_view(
                            4096,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                            shape=(64, 16),
                        )
                        value_6 = T.tensor_view(
                            T.ptr_of(value_10),
                            layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                            shape=(64, 16),
                        )
                        lhs_view = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 16]),
                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(64, 16),
                        )
                        value_3_2 = T.tensor_view(
                            0,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 0, 1))) | Swizzle(3, 4, 3),
                            shape=(16, 128),
                        )
                        value_7 = T.tensor_view(
                            T.ptr_of(value_3_2),
                            layout=Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 0, 1))) | Swizzle(3, 4, 3),
                            shape=(16, 128),
                        )
                        rhs_view = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 16, 0:0 + 128]),
                            layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 0, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 128),
                        )
                        T.tiled_mma(
                            acc,
                            lhs_view,
                            rhs_view,
                            atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
            value_8 = T.tensor_view(
                T.ptr_of(acc),
                layout=((4 @ scope.d1, 2, 8 @ scope.d2, 2 @ scope.d0, 8, 4 @ scope.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                shape=(64, 128),
            )
            value_9 = T.tensor_view(T.ptr_of(out), layout=(64, 128), shape=(64, 128))
            window = T.tensor_view(
                T.ptr_of(value_9[0:0 + 64, 0:0 + 128]), layout=(64, 128), shape=(64, 128)
            )
            T.copy(value_8, window)
