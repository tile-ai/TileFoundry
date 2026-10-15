# analysis target=nvidia.h200_sxm module=SCHED_GDN_NORM_T1024 function=gdn_norm topology=cta wave=132/6144
# selection requested=memory executed=memory
# memory traffic=gmem:r6.25KB/w12.00MB@logical,r37.50MB/w72.02GB@total,r6.25KB/w12.00MB@cta,r2.05KB/w12.00MB@thread;rmem:r70.96KB/w58.88KB@logical,r425.79MB/w353.30MB@total,r70.96KB/w58.88KB@cta,r716B/w536B@thread footprint=gamma:256B footprint-precision=lower_bound peak=gmem:68.50MB;rmem:8.05KB persistent=gmem:44.50MB
#   buffer=gamma holds=256B time=none space=cta.t,cta.hb reuse=32.75KB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_norm(
    o: Tensor[(1024, 48, 128), "bf16"], proj: Tensor[(1024, 16640), "bf16"], gamma: Tensor[(128,), "bf16"], scratch: Tensor[(1024,), "bf16"], out: Tensor[(1024, 48, 128), "bf16"]
):
    with Mesh((Topology("cta", 6144),), layout=(1024, 6), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 1024]), layout=(8, 128), shape=(8, 128))
        y = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "bf16", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        n2 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "bf16", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        nb = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "bf16", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        rf = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "bf16", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        rr = T.alloc_tensor(
            tensor_type=Tensor[(8, 1), "f32", ((4, 2, 1, 1), (2, 1, 0, 0)), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(8, 1), "f32", ((4, 2, 1, 1), (2, 1, 0, 0)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(8, 1), "f32", ((4, 2, 1, 1), (2, 1, 0, 0)), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(8, 1), "f32", ((4, 2, 1, 1), (2, 1, 0, 0)), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        gm = T.alloc_tensor(
            tensor_type=Tensor[(1, 128), "f32", ((1, 16, 8), (0, 0, 1)), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(1, 128), "bf16", ((1, 16, 8), (0, 0, 1)), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        zf = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "bf16", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(8, 128), "f32", ((4, 2, 16, 8), (0, 0, 0, 1)), "rmem"]
        )
        with Mesh(
            (Topology("thread", 128),), layout=(4, 2, 16), names=("d0", "d1", "d2")
        ) as scope:
            tile = T.tensor_view(
                T.ptr_of(o[cta.d0:cta.d0 + 1, cta.d1 * 8:cta.d1 * 8 + 8, 0:0 + 128]),
                layout=((1, 8, 128), (6144, 128, 1)),
                shape=(1, 8, 128),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[0:0 + 1, 0:0 + 8, 0:0 + 128]), layout=(8, 128), shape=(8, 128)
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(tile_1, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(rf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                lhs_frame = T.tensor_view(
                    T.ptr_of(rf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(rf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame, rhs_frame, dst_frame_2, kind=BinaryKind.MUL)
                src_frame_1 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 16 @ threads.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads.d0, 2 @ threads.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                T.reduce(
                    src_frame_1,
                    dst_frame_3,
                    axes=(-1,),
                    keepdim=True,
                    kind=ReduceKind.SUM,
                )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 0.0078125)
            with scope as threads_4:
                lhs_frame_1 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_4.d0, 2 @ threads_4.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_4.d0, 2 @ threads_4.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                T.binary(lhs_frame_1, scalar, dst_frame_4, kind=BinaryKind.MUL)
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1e-06)
            with scope as threads_5:
                lhs_frame_2 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                T.binary(lhs_frame_2, scalar_1, dst_frame_5, kind=BinaryKind.ADD)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(rr[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                T.unary(src_frame_2, dst_frame_6, kind=UnaryKind.RSQRT)
                lhs_frame_3 = T.tensor_view(
                    T.ptr_of(rf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                rhs_frame_1 = T.tensor_view(
                    T.ptr_of(rr[0:0 + 8, 0:0 + 1]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 1, 1), (2, 1, 0, 0)),
                    shape=(8, 1),
                )
                dst_frame_7 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame_3, rhs_frame_1, dst_frame_7, kind=BinaryKind.MUL)
                src_frame_3 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_8 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_3, dst_frame_8, dtype="bf16")
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_9 = T.tensor_view(
                    T.ptr_of(nb[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_5.d0, 2 @ threads_5.d1, 16 @ threads_5.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_4, dst_frame_9, dtype="f32")
            tile_2 = T.tensor_view(T.ptr_of(gamma[0:0 + 128]), layout=(1, 128), shape=(1, 128))
            with scope as threads_10:
                dst_frame_10 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 1, 0:0 + 128]),
                    layout=((1, 16 @ threads_10.d2, 8), (0, 0, 1)),
                    shape=(1, 128),
                )
                T.copy(tile_2, dst_frame_10)
                src_frame_5 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 1, 0:0 + 128]),
                    layout=((1, 16 @ threads_10.d2, 8), (0, 0, 1)),
                    shape=(1, 128),
                )
                dst_frame_11 = T.tensor_view(
                    T.ptr_of(gm[0:0 + 1, 0:0 + 128]),
                    layout=((1, 16 @ threads_10.d2, 8), (0, 0, 1)),
                    shape=(1, 128),
                )
                T.cast(src_frame_5, dst_frame_11, dtype="f32")
                lhs_frame_4 = T.tensor_view(
                    T.ptr_of(nb[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                rhs_frame_2 = T.tensor_view(
                    T.ptr_of(gm[0:0 + 1, 0:0 + 128]),
                    layout=((1, 16 @ threads_10.d2, 8), (0, 0, 1)),
                    shape=(1, 128),
                )
                dst_frame_12 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame_4, rhs_frame_2, dst_frame_12, kind=BinaryKind.MUL)
                src_frame_6 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_13 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_6, dst_frame_13, dtype="bf16")
                src_frame_7 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_14 = T.tensor_view(
                    T.ptr_of(n2[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_10.d0, 2 @ threads_10.d1, 16 @ threads_10.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_7, dst_frame_14, dtype="f32")
            tile_3 = T.tensor_view(
                T.ptr_of(proj[cta.d0:cta.d0 + 1, 10240 + ((cta.d1 * 8) * 128):10240 + ((cta.d1 * 8) * 128) + 1024]),
                layout=((1, 1024), (16640, 1)),
                shape=(1, 1024),
            )
            tile_4 = T.tensor_view(
                T.ptr_of(tile_3[0:0 + 1, 0:0 + 1024]), layout=(8, 128), shape=(8, 128)
            )
            with scope as threads_15:
                dst_frame_15 = T.tensor_view(
                    T.ptr_of(value_14[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(tile_4, dst_frame_15)
                src_frame_8 = T.tensor_view(
                    T.ptr_of(value_14[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_16 = T.tensor_view(
                    T.ptr_of(zf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_8, dst_frame_16, dtype="f32")
                src_frame_9 = T.tensor_view(
                    T.ptr_of(zf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_17 = T.tensor_view(
                    T.ptr_of(value_17[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.unary(src_frame_9, dst_frame_17, kind=UnaryKind.NEG)
                src_frame_10 = T.tensor_view(
                    T.ptr_of(value_17[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_18 = T.tensor_view(
                    T.ptr_of(value_16[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_15.d0, 2 @ threads_15.d1, 16 @ threads_15.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.unary(src_frame_10, dst_frame_18, kind=UnaryKind.EXP)
            scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_2, 1.0)
            with scope as threads_19:
                lhs_frame_5 = T.tensor_view(
                    T.ptr_of(value_16[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_19 = T.tensor_view(
                    T.ptr_of(value_15[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame_5, scalar_2, dst_frame_19, kind=BinaryKind.ADD)
                lhs_frame_6 = T.tensor_view(
                    T.ptr_of(zf[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                rhs_frame_3 = T.tensor_view(
                    T.ptr_of(value_15[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_20 = T.tensor_view(
                    T.ptr_of(value_13[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame_6, rhs_frame_3, dst_frame_20, kind=BinaryKind.DIV)
                lhs_frame_7 = T.tensor_view(
                    T.ptr_of(n2[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                rhs_frame_4 = T.tensor_view(
                    T.ptr_of(value_13[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_21 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.binary(lhs_frame_7, rhs_frame_4, dst_frame_21, kind=BinaryKind.MUL)
                src_frame_11 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                dst_frame_22 = T.tensor_view(
                    T.ptr_of(y[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.cast(src_frame_11, dst_frame_22, dtype="bf16")
                src_frame_12 = T.tensor_view(
                    T.ptr_of(y[0:0 + 8, 0:0 + 128]),
                    layout=((4 @ threads_19.d0, 2 @ threads_19.d1, 16 @ threads_19.d2, 8), (0, 0, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(src_frame_12, value_1)
            tile_5 = T.tensor_view(
                T.ptr_of(value_1[0:0 + 8, 0:0 + 128]), layout=(1, 8, 128), shape=(1, 8, 128)
            )
            window = T.tensor_view(
                T.ptr_of(out[cta.d0:cta.d0 + 1, cta.d1 * 8:cta.d1 * 8 + 8, 0:0 + 128]),
                layout=((1, 8, 128), (6144, 128, 1)),
                shape=(1, 8, 128),
            )
            T.copy(tile_5, window)
