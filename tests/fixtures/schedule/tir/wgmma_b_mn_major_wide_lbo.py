# analysis target=nvidia.h200_sxm module=MnMajorB function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r10.00KB/w64.00KB@logical,r10.00KB/w64.00KB@total,r10.00KB/w64.00KB@cta,r10.00KB/w32.25KB@thread;rmem:r64.03KB/w64.00KB@logical,r64.03KB/w64.00KB@total,r64.03KB/w64.00KB@cta,r544B/w512B@thread;smem:r6.00KB/w10.00KB@logical,r6.00KB/w10.00KB@total,r6.00KB/w10.00KB@cta,r1.06KB/w10.00KB@thread footprint=a:2.00KB;v:8.00KB;v9:24:32.00KB footprint-precision=exact peak=gmem:42.00KB;rmem:32.00KB;smem:10.00KB persistent=gmem:10.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 16), "f16"], v: Tensor[(32, 128), "f16"], out: Tensor[(64, 128), "f32"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh(
            (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
        ) as scope:
            T.fill(out, 0.0)
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 128),
                    "f32",
                    ((4 @ scope.d0, 2, 8 @ scope.d1, 16, 4 @ scope.d2, 2), (16, 8, 1, 512, 128, 64)),
                    "rmem",
                ]
            )
            T.fill(acc, 0.0)
            value = T.tensor_view(
                0,
                dtype="f16",
                storage="smem",
                layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                shape=(64, 16),
            )
            T.copy(a, value)
            value_1 = T.tensor_view(
                2048,
                dtype="f16",
                storage="smem",
                layout=Layout((32, (2, 64)), (64, (2048, 1))) | Swizzle(3, 4, 3),
                shape=(32, 128),
            )
            T.copy(v, value_1)
            for o_m in range(0, 64, 64):
                for o_n in range(0, 128, 128):
                    for o_k in range(0, 16, 16):
                        value_2 = T.tensor_view(
                            0,
                            dtype="f16",
                            storage="smem",
                            layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                            shape=(64, 16),
                        )
                        lhs_view = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 64, 0:0 + 16]),
                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(64, 16),
                        )
                        value_1_2 = T.tensor_view(
                            2048,
                            dtype="f16",
                            storage="smem",
                            layout=Layout((32, (2, 64)), (64, (2048, 1))) | Swizzle(3, 4, 3),
                            shape=(32, 128),
                        )
                        tile = T.tensor_view(
                            T.ptr_of(value_1_2[0:0 + 16, 0:0 + 128]),
                            layout=Layout((16, (2, 64)), (64, (2048, 1))) | Swizzle(3, 4, 3),
                            shape=(16, 128),
                        )
                        tile_1 = T.tensor_view(
                            T.ptr_of(tile[0:0 + 16, 0:0 + 128]),
                            layout=Layout(((2, 8), (2, 64)), ((512, 64), (2048, 1))) | Swizzle(3, 4, 3),
                            shape=(16, 128),
                        )
                        rhs_view = T.tensor_view(
                            T.ptr_of(tile_1[0:0 + 16, 0:0 + 128]),
                            layout=(Layout(((2, 8), (2, 64)), ((512, 64), (2048, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 128),
                        )
                        T.tiled_mma(
                            acc,
                            lhs_view,
                            rhs_view,
                            atom=T.cuda.sm90.Wgmma(n=128, dtype="f16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
            window = T.tensor_view(
                T.ptr_of(out[0:0 + 64, 0:0 + 128]), layout=(64, 128), shape=(64, 128)
            )
            T.copy(acc, window)
