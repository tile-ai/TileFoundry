from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 16), "bf16"], weight: Tensor[(16, 16), "bf16"], scratch: Tensor[(256,), "bf16"], out: Tensor[(64, 16), "f32"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh(
            (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
        ) as scope:
            T.fill(out, 0.0)
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 16),
                    "f32",
                    ((4 @ scope.d0, 2, 8 @ scope.d1, 2, 4 @ scope.d2, 2), (16, 8, 1, 512, 128, 64)),
                    "rmem",
                ]
            )
            T.fill(acc, 0.0)
            value = T.tensor_view(T.ptr_of(a), layout=(64, 16), shape=(64, 16))
            lhs_dst = T.tensor_view(
                512,
                dtype="bf16",
                storage="smem",
                layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                shape=(64, 16),
            )
            value_1 = T.tensor_view(
                T.ptr_of(lhs_dst),
                layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                shape=(64, 16),
            )
            T.copy(value, value_1)
            value_2 = T.tensor_view(T.ptr_of(weight), layout=(16, 16), shape=(16, 16))
            tile = T.tensor_view(
                T.ptr_of(value_2[0:0 + 16, 0:0 + 16]),
                layout=((16, 16), (1, 16)),
                shape=(16, 16),
            )
            rhs_kn_1 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 256]), layout=(16, 16), shape=(16, 16)
            )
            T.copy(tile, rhs_kn_1)
            rhs_dst = T.tensor_view(
                0,
                dtype="bf16",
                storage="smem",
                layout=Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))),
                shape=(16, 16),
            )
            value_3 = T.tensor_view(
                T.ptr_of(rhs_dst),
                layout=Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))),
                shape=(16, 16),
            )
            T.copy(rhs_kn_1, value_3)
            for o_m in range(0, 64, 64):
                for o_n in range(0, 16, 16):
                    for o_k in range(0, 16, 16):
                        lhs_dst_2 = T.tensor_view(
                            512,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                            shape=(64, 16),
                        )
                        value_4 = T.tensor_view(
                            T.ptr_of(lhs_dst_2),
                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                            shape=(64, 16),
                        )
                        lhs_view = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 64, 0:0 + 16]),
                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                            shape=(64, 16),
                        )
                        rhs_dst_2 = T.tensor_view(
                            0,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))),
                            shape=(16, 16),
                        )
                        value_5 = T.tensor_view(
                            T.ptr_of(rhs_dst_2),
                            layout=Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))),
                            shape=(16, 16),
                        )
                        rhs_view = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 16, 0:0 + 16]),
                            layout=Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))) @ {B()},
                            shape=(16, 16),
                        )
                        T.tiled_mma(
                            acc,
                            lhs_view,
                            rhs_view,
                            atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                        )
            value_6 = T.tensor_view(
                T.ptr_of(acc),
                layout=((4 @ scope.d0, 2, 8 @ scope.d1, 2, 4 @ scope.d2, 2), (16, 8, 1, 512, 128, 64)),
                shape=(64, 16),
            )
            value_7 = T.tensor_view(T.ptr_of(out), layout=(64, 16), shape=(64, 16))
            window = T.tensor_view(
                T.ptr_of(value_7[0:0 + 64, 0:0 + 16]), layout=(64, 16), shape=(64, 16)
            )
            T.copy(value_6, window)
