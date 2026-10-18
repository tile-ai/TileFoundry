# analysis target=nvidia.h200_sxm module=Final function=final topology=cta wave=132/7760
# selection requested=memory executed=memory
# memory traffic=gmem:r340.01KB/w485.06KB@logical,r2.52GB/w3.59GB@total,r340.01KB/w485.06KB@cta,r1.34KB/w485.01KB@thread;rmem:r2.40MB/w2.38MB@logical,r18.21GB/w18.06GB@total,r2.40MB/w2.38MB@cta,r10.63KB/w10.42KB@thread;smem:r10.00KB/w10.00KB@logical,r75.78MB/w75.78MB@total,r10.00KB/w10.00KB@cta,r320B/w40B@thread footprint=<value 0>[0]:8B;gamma_final:10.00KB;hidden:10.00KB;v46:86:8.25KB;w_head:41.25MB footprint-precision=upper_bound peak=gmem:2.38GB;rmem:1.25MB;smem:10.00KB persistent=gmem:2.38GB
#   buffer=hidden holds=41.28MB time=none space=cta.b reuse=1.28MB fits=yes precision=upper_bound
#   buffer=gamma_final holds=41.28MB time=none space=cta.b reuse=1.28MB fits=yes precision=upper_bound
#   buffer=<value 0>[0] holds=41.28MB time=none space=cta.b reuse=1.02KB fits=yes precision=upper_bound
#   error="rmem placement peak 1.25MB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def final(
    hidden: Tensor[(1, 1024, 5120), "bf16"], last: Tensor[(1,), "i64"], gamma_final: ConstTensor[(5120,), "bf16"], w_head: ConstTensor[(248320, 5120), "bf16"], out: Tensor[(62080, 4, 1, 1, 1), "bf16"]
):
    with Mesh((Topology("cta", 7760),), layout=(7760,), names=("b",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(256,), names=("t",)) as scope:
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            value_2 = T.tensor_view(T.ptr_of(last), layout=(1,), shape=(1,))
            value_3 = T.tensor_view(T.ptr_of(value_2), layout=(1,), shape=(1,))
            value_4 = T.tensor_view(T.ptr_of(value_3), layout=(), shape=())
            value_5 = T.tensor_view(
                T.ptr_of(hidden), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
            )
            value_6 = T.tensor_view(
                T.ptr_of(value_5), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            value_7 = T.tensor_view(
                T.ptr_of(value_6[value_4:value_4 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                shape=(1, 5, 256, 4),
            )
            value_8 = T.tensor_view(
                T.ptr_of(value_1), layout=(1, 5, 256 @ scope.t, 4), shape=(1, 5, 256, 4)
            )
            dst_view = T.tensor_view(
                T.ptr_of(value_8[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.copy(value_7, dst_view)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            src_view = T.tensor_view(
                T.ptr_of(value_8[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.cast(src_view, x32, dtype="f32")
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5, 256, 4), "f32", ((1, 5, 256 @ scope.t, 4), (0, 4, 0, 1)), "rmem"
                ]
            )
            value_11 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.unary(x32, value_11, kind=UnaryKind.SQUARE)
            value_12 = T.tensor_view(
                T.ptr_of(value_10),
                layout=((1, 5, 256 @ scope.t, 4), (0, 4, 0, 1)),
                shape=(1, 5, 256, 4),
            )
            dst_view_1 = T.tensor_view(
                T.ptr_of(value_12[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=((1, 5, 256 @ scope.t, 4), (0, 4, 0, 1)),
                shape=(1, 5, 256, 4),
            )
            T.copy(value_11, dst_view_1)
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 1, 1),
                    "f32",
                    ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)) @ {B()},
                    "rmem",
                ]
            )
            src_view_1 = T.tensor_view(
                T.ptr_of(value_12[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=((1, 5, 256 @ scope.t, 4), (0, 4, 0, 1)),
                shape=(1, 5, 256, 4),
            )
            T.reduce(src_view_1, value_13, axes=(1, 2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_14 = T.tensor_view(T.ptr_of(value_9), layout=(1, 1, 1, 1), shape=(1, 1, 1, 1))
            T.copy(value_13, value_14)
            value_15 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(value_14, scalar, value_15, kind=BinaryKind.ADD)
            value_16 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            T.unary(value_15, value_16, kind=UnaryKind.RSQRT)
            value_17 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.binary(x32, value_16, value_17, kind=BinaryKind.MUL)
            value_18 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            value_19 = T.tensor_view(T.ptr_of(gamma_final), layout=(5120,), shape=(5120,))
            value_20 = T.tensor_view(
                T.ptr_of(value_19),
                layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                shape=(1, 5, 256, 4),
            )
            value_21 = T.tensor_view(
                T.ptr_of(value_18), layout=(1, 5, 256 @ scope.t, 4), shape=(1, 5, 256, 4)
            )
            dst_view_2 = T.tensor_view(
                T.ptr_of(value_21[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.copy(value_20, dst_view_2)
            value_22 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            src_view_2 = T.tensor_view(
                T.ptr_of(value_21[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.cast(src_view_2, value_22, dtype="f32")
            value_23 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_22, scalar_1, value_23, kind=BinaryKind.ADD)
            value_24 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.binary(value_17, value_23, value_24, kind=BinaryKind.MUL)
            xn = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.cast(value_24, xn, dtype="bf16")
            value = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_25 = T.tensor_view(
                T.ptr_of(value), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            T.copy(xn, value_25)
        with Mesh((Topology("thread", 256),), layout=(8, 32), names=("w", "l")) as scope_1:
            value_26 = T.tensor_view(
                T.ptr_of(out), layout=(62080, 4, 1, 1, 1), shape=(62080, 4, 1, 1, 1)
            )
            T.fill(value_26, 0.0)
            value_27 = T.alloc_tensor(
                tensor_type=Tensor[(8, 4, 1, 1, 1), "f32", (8 @ scope_1.w, 4, 1, 1, 1), "rmem"]
            )
            value_28 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8),
                    "f32",
                    ((8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), (0, 160, 8, 0, 1)),
                    "rmem",
                ]
            )
            value_29 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8), "bf16", (8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), "rmem"
                ]
            )
            value_30 = T.tensor_view(
                T.ptr_of(w_head), layout=(248320, 5120), shape=(248320, 5120)
            )
            value_31 = T.tensor_view(
                T.ptr_of(value_30), layout=(62080, 4, 20, 32, 8), shape=(62080, 4, 20, 32, 8)
            )
            value_32 = T.tensor_view(
                T.ptr_of(value_31[(cta.b * 32) // 4:(cta.b * 32) // 4 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(8, 4, 20, 32, 8),
                shape=(8, 4, 20, 32, 8),
            )
            value_33 = T.tensor_view(
                T.ptr_of(value_29),
                layout=(8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8),
                shape=(8, 4, 20, 32, 8),
            )
            dst_view_3 = T.tensor_view(
                T.ptr_of(value_33[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=(8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8),
                shape=(8, 4, 20, 32, 8),
            )
            T.copy(value_32, dst_view_3)
            value_34 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8), "f32", (8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), "rmem"
                ]
            )
            src_view_3 = T.tensor_view(
                T.ptr_of(value_33[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=(8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8),
                shape=(8, 4, 20, 32, 8),
            )
            T.cast(src_view_3, value_34, dtype="f32")
            value_35 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 20, 32, 8), "bf16", (1, 1, 20, 32 @ scope_1.l, 8), "rmem"
                ]
            )
            value_47 = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_36 = T.tensor_view(
                T.ptr_of(value_47), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_37 = T.tensor_view(
                T.ptr_of(value_36), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_38 = T.tensor_view(
                T.ptr_of(value_37), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_39 = T.tensor_view(
                T.ptr_of(value_38),
                layout=((1, 1, 20, 32, 8), (0, 0, 256, 8, 1)),
                shape=(1, 1, 20, 32, 8),
            )
            value_40 = T.tensor_view(
                T.ptr_of(value_35),
                layout=(1, 1, 20, 32 @ scope_1.l, 8),
                shape=(1, 1, 20, 32, 8),
            )
            dst_view_4 = T.tensor_view(
                T.ptr_of(value_40[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=(1, 1, 20, 32 @ scope_1.l, 8),
                shape=(1, 1, 20, 32, 8),
            )
            T.copy(value_39, dst_view_4)
            value_41 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 20, 32, 8), "f32", (1, 1, 20, 32 @ scope_1.l, 8), "rmem"
                ]
            )
            src_view_4 = T.tensor_view(
                T.ptr_of(value_40[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=(1, 1, 20, 32 @ scope_1.l, 8),
                shape=(1, 1, 20, 32, 8),
            )
            T.cast(src_view_4, value_41, dtype="f32")
            value_42 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8), "f32", (8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), "rmem"
                ]
            )
            T.binary(value_34, value_41, value_42, kind=BinaryKind.MUL)
            value_43 = T.tensor_view(
                T.ptr_of(value_28),
                layout=((8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), (0, 160, 8, 0, 1)),
                shape=(8, 4, 20, 32, 8),
            )
            dst_view_5 = T.tensor_view(
                T.ptr_of(value_43[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=((8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), (0, 160, 8, 0, 1)),
                shape=(8, 4, 20, 32, 8),
            )
            T.copy(value_42, dst_view_5)
            value_44 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 1, 1, 1),
                    "f32",
                    ((8 @ scope_1.w, 4, 1, 1, 1, 1, 1), (4, 1, 0, 0, 0, 0, 0)),
                    "rmem",
                ]
            )
            src_view_5 = T.tensor_view(
                T.ptr_of(value_43[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=((8 @ scope_1.w, 4, 20, 32 @ scope_1.l, 8), (0, 160, 8, 0, 1)),
                shape=(8, 4, 20, 32, 8),
            )
            T.reduce(src_view_5, value_44, axes=(2, 3, 4), keepdim=True, kind=ReduceKind.SUM)
            value_45 = T.tensor_view(
                T.ptr_of(value_27), layout=(8 @ scope_1.w, 4, 1, 1, 1), shape=(8, 4, 1, 1, 1)
            )
            dst_view_6 = T.tensor_view(
                T.ptr_of(value_45[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(8 @ scope_1.w, 4, 1, 1, 1),
                shape=(8, 4, 1, 1, 1),
            )
            T.copy(value_44, dst_view_6)
            logit = T.alloc_tensor(
                tensor_type=Tensor[(8, 4, 1, 1, 1), "bf16", (8 @ scope_1.w, 4, 1, 1, 1), "rmem"]
            )
            src_view_6 = T.tensor_view(
                T.ptr_of(value_45[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(8 @ scope_1.w, 4, 1, 1, 1),
                shape=(8, 4, 1, 1, 1),
            )
            T.cast(src_view_6, logit, dtype="bf16")
            value_46 = T.tensor_view(
                T.ptr_of(out), layout=(62080, 4, 1, 1, 1), shape=(62080, 4, 1, 1, 1)
            )
            window = T.tensor_view(
                T.ptr_of(value_46[(cta.b * 32) // 4:(cta.b * 32) // 4 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(8, 4, 1, 1, 1),
                shape=(8, 4, 1, 1, 1),
            )
            T.copy(logit, window)
