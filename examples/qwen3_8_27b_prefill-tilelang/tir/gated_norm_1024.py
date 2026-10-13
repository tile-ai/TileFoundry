# analysis target=nvidia.h200_sxm module=GatedNorm function=gated_norm topology=cta wave=132/3072
# selection requested=memory executed=memory
# memory traffic=gmem:r8.25KB/w12.00MB@logical,r24.75MB/w36.01GB@total,r8.25KB/w12.00MB@cta,r48B/w12.00MB@thread;rmem:r141.15KB/w125.02KB@logical,r423.45MB/w375.07MB@total,r141.15KB/w125.02KB@cta,r776B/w584B@thread footprint=gamma_gdn:256B footprint-precision=lower_bound peak=gmem:48.00MB;rmem:16.08KB persistent=gmem:24.00MB
#   buffer=gamma_gdn holds=256B time=none space=cta.b reuse=32.75KB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gated_norm(
    core: Tensor[(1, 1024, 48, 128), "bf16"], z: Tensor[(1, 1024, 6144), "bf16", ((1, 1024, 6144), (17039360, 16640, 1))], gamma_gdn: Tensor[(128,), "bf16"], out: Tensor[(1024, 48, 16, 8), "bf16"]
):
    with Mesh((Topology("cta", 3072),), layout=(3072,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        y = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "bf16", (1, 16, 16, 8), "rmem"])
        value = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        nw = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "bf16", (1, 16, 16, 8), "rmem"])
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        n = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "bf16", (1, 16, 16, 8), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        x32 = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"])
        x = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "bf16", (1, 16, 16, 8), "rmem"])
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16, 1, 1), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16, 1, 1), "rmem"]
        )
        ms = T.alloc_tensor(tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16, 1, 1), "rmem"])
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (1, 16, 1, 1), "f32", ((1, 16, 1, 1, 1), (0, 1, 0, 0, 0)), "rmem"
            ]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", ((1, 16, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16, 8), "rmem"]
        )
        g = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16, 8), "rmem"])
        gate = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"])
        z32 = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"])
        zz = T.alloc_tensor(tensor_type=Tensor[(1, 16, 16, 8), "bf16", (1, 16, 16, 8), "rmem"])
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(1, 16, 16, 8), "f32", (1, 16, 16, 8), "rmem"]
        )
        with Mesh((Topology("thread", 256),), layout=(16, 16), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(core[0:0 + 1, 0:0 + 1024, 0:0 + 48, 0:0 + 128]),
                layout=(1024, 48, 16, 8),
                shape=(1024, 48, 16, 8),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[cta.d0 // 3:cta.d0 // 3 + 1, (cta.d0 % 3) * 16:(cta.d0 % 3) * 16 + 16, 0:0 + 16, 0:0 + 8]),
                layout=((1, 16, 16, 8), (6144, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.copy(tile_1, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                src_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.unary(src_frame_1, dst_frame_2, kind=UnaryKind.SQUARE)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads.d0, 16 @ threads.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=((1, 16 @ threads.d0, 16 @ threads.d1, 8), (0, 0, 0, 1)),
                    shape=(1, 16, 16, 8),
                )
                T.copy(src_frame_2, dst_frame_3)
                src_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=((1, 16 @ threads.d0, 16 @ threads.d1, 8), (0, 0, 0, 1)),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 16 @ threads.d0, 1, 1, 1), (0, 1, 0, 0, 0)),
                    shape=(1, 16, 1, 1),
                )
                T.reduce(
                    src_frame_3,
                    dst_frame_4,
                    axes=(2, 3),
                    keepdim=True,
                    kind=ReduceKind.MEAN,
                )
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 16 @ threads.d0, 1, 1, 1), (0, 1, 0, 0, 0)),
                    shape=(1, 16, 1, 1),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(ms[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                T.copy(src_frame_4, dst_frame_5)
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            with scope as threads_6:
                lhs_frame = T.tensor_view(
                    T.ptr_of(ms[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads_6.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads_6.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                T.binary(lhs_frame, scalar, dst_frame_6, kind=BinaryKind.ADD)
                src_frame_5 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads_6.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                dst_frame_7 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads_6.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                T.unary(src_frame_5, dst_frame_7, kind=UnaryKind.RSQRT)
                lhs_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 16 @ threads_6.d0, 1, 1),
                    shape=(1, 16, 1, 1),
                )
                dst_frame_8 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.binary(lhs_frame_1, rhs_frame, dst_frame_8, kind=BinaryKind.MUL)
                src_frame_6 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_9 = T.tensor_view(
                    T.ptr_of(n[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_6, dst_frame_9, dtype="bf16")
                src_frame_7 = T.tensor_view(
                    T.ptr_of(n[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_10 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_6.d0, 16 @ threads_6.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_7, dst_frame_10, dtype="f32")
            tile_2 = T.tensor_view(
                T.ptr_of(gamma_gdn[0:0 + 128]), layout=(1, 1, 16, 8), shape=(1, 1, 16, 8)
            )
            with scope as threads_11:
                dst_frame_11 = T.tensor_view(
                    T.ptr_of(g[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ threads_11.d1, 8),
                    shape=(1, 1, 16, 8),
                )
                T.copy(tile_2, dst_frame_11)
                src_frame_8 = T.tensor_view(
                    T.ptr_of(g[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ threads_11.d1, 8),
                    shape=(1, 1, 16, 8),
                )
                dst_frame_12 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ threads_11.d1, 8),
                    shape=(1, 1, 16, 8),
                )
                T.cast(src_frame_8, dst_frame_12, dtype="f32")
                lhs_frame_2 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                rhs_frame_1 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ threads_11.d1, 8),
                    shape=(1, 1, 16, 8),
                )
                dst_frame_13 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.binary(lhs_frame_2, rhs_frame_1, dst_frame_13, kind=BinaryKind.MUL)
                src_frame_9 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_14 = T.tensor_view(
                    T.ptr_of(nw[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_9, dst_frame_14, dtype="bf16")
                src_frame_10 = T.tensor_view(
                    T.ptr_of(nw[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_15 = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_11.d0, 16 @ threads_11.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_10, dst_frame_15, dtype="f32")
            tile_3 = T.tensor_view(
                T.ptr_of(z[0:0 + 1, 0:0 + 1024, 0:0 + 6144]),
                layout=(1024, 48, 16, 8),
                shape=(1024, 48, 16, 8),
            )
            tile_4 = T.tensor_view(
                T.ptr_of(tile_3[cta.d0 // 3:cta.d0 // 3 + 1, (cta.d0 % 3) * 16:(cta.d0 % 3) * 16 + 16, 0:0 + 16, 0:0 + 8]),
                layout=(1, 16, 16, 8),
                shape=(1, 16, 16, 8),
            )
            with scope as threads_16:
                dst_frame_16 = T.tensor_view(
                    T.ptr_of(zz[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.copy(tile_4, dst_frame_16)
                src_frame_11 = T.tensor_view(
                    T.ptr_of(zz[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_17 = T.tensor_view(
                    T.ptr_of(z32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_11, dst_frame_17, dtype="f32")
                src_frame_12 = T.tensor_view(
                    T.ptr_of(z32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_18 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.unary(src_frame_12, dst_frame_18, kind=UnaryKind.NEG)
                src_frame_13 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_19 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_16.d0, 16 @ threads_16.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.unary(src_frame_13, dst_frame_19, kind=UnaryKind.EXP)
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            with scope as threads_20:
                lhs_frame_3 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_20 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.binary(lhs_frame_3, scalar_1, dst_frame_20, kind=BinaryKind.ADD)
                lhs_frame_4 = T.tensor_view(
                    T.ptr_of(z32[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                rhs_frame_2 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_21 = T.tensor_view(
                    T.ptr_of(gate[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.binary(lhs_frame_4, rhs_frame_2, dst_frame_21, kind=BinaryKind.DIV)
                lhs_frame_5 = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                rhs_frame_3 = T.tensor_view(
                    T.ptr_of(gate[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_22 = T.tensor_view(
                    T.ptr_of(value[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.binary(lhs_frame_5, rhs_frame_3, dst_frame_22, kind=BinaryKind.MUL)
                src_frame_14 = T.tensor_view(
                    T.ptr_of(value[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                dst_frame_23 = T.tensor_view(
                    T.ptr_of(y[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                T.cast(src_frame_14, dst_frame_23, dtype="bf16")
                y_view = T.tensor_view(
                    T.ptr_of(y[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 16 @ threads_20.d0, 16 @ threads_20.d1, 8),
                    shape=(1, 16, 16, 8),
                )
                window = T.tensor_view(
                    T.ptr_of(out[cta.d0 // 3:cta.d0 // 3 + 1, (cta.d0 % 3) * 16:(cta.d0 % 3) * 16 + 16, 0:0 + 16, 0:0 + 8]),
                    layout=((1, 16, 16, 8), (6144, 128, 8, 1)),
                    shape=(1, 16, 16, 8),
                )
                T.copy(y_view, window)
