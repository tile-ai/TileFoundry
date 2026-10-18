# analysis target=nvidia.h200_sxm module=InProj function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r167.81MB/w40.62MB@logical,r31.94GB/w5.24GB@total,r247.81MB/w40.62MB@cta,r247.81MB/w40.62MB@thread;rmem:r662.27MB/w654.19MB@logical,r85.38GB/w85.36GB@total,r662.35MB/w662.19MB@cta,r2.75MB/w2.59MB@thread;smem:r247.81MB/w167.81MB@logical,r31.94GB/w31.94GB@total,r247.81MB/w247.81MB@cta,r45.96MB/w243.77MB@thread footprint=v32:426:64.00KB;v34:427:8.25MB;w:544.00KB;x:128.00KB footprint-precision=exact peak=gmem:205.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:172.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 5120), "bf16"], w: Tensor[(16640, 5120), "bf16"], out: Tensor[(1024, 16640), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_3_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for g in range(0, 1, 1):
                for bn in range(0, 65, 1):
                    for mi in range(((cta.d0 + (8 * g)) - (8 * bn)) - (132 * (((cta.d0 + (8 * g)) - (8 * bn)) // 132)), 8, 132):
                        acc = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "f32",
                                ShardLayout(
                                    layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    attrs=(S(0), S(1), S(3), S(5)),
                                    mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                                ),
                                "rmem",
                            ]
                        )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_1:
                            T.fill(acc, 0.0)
                        for k in range(0, 5120, 64):
                            with scope[2:, :32] as scope_2:
                                value_1 = T.tensor_view(
                                    T.ptr_of(x[0:0 + 1, ((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                value_2 = T.tensor_view(
                                    T.ptr_of(value_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_1, value_2)
                                value_4 = T.tensor_view(
                                    T.ptr_of(w[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_3_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                T.copy_async_tensor(value_4, value_5)
                            with Mesh(
                                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_3:
                                for o_m in range(0, 128, 128):
                                    for o_n in range(0, 256, 256):
                                        for o_k in range(0, 64, 64):
                                            value_6 = T.tensor_view(
                                                T.ptr_of(acc),
                                                layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                                shape=(128, 256),
                                            )
                                            value_7 = T.tensor_view(
                                                T.ptr_of(value_stages[stage_counter % 3]),
                                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                                shape=(128, 64),
                                            )
                                            lhs_view = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            value_8 = T.tensor_view(
                                                T.ptr_of(value_3_stages[stage_counter % 3]),
                                                layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                                shape=(64, 256),
                                            )
                                            rhs_view = T.tensor_view(
                                                T.ptr_of(value_8[o_k:o_k + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view,
                                                rhs_view,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_1,
                                                rhs_view_1,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_2,
                                                rhs_view_2,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_3,
                                                rhs_view_3,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                            T.binary(
                                stage_counter,
                                stage_one,
                                stage_counter,
                                kind=BinaryKind.ADD,
                            )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_4:
                            value_10 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_11 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(value_11, value_10, dtype="bf16")
                            value_9 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_12 = T.tensor_view(
                                T.ptr_of(value_9),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(value_10, value_12)
                        with scope[:1, :32] as scope_5:
                            value_9_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_13 = T.tensor_view(
                                T.ptr_of(value_9_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_14 = T.tensor_view(
                                T.ptr_of(out[128 * mi:128 * mi + 1, 256 * bn:256 * bn + 1]),
                                layout=((128, 256), (16640, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(value_13, value_14)
