# analysis target=nvidia.h200_sxm module=GatedNorm function=gated_norm topology=cta wave=132/3072
# selection requested=memory executed=memory
# memory traffic=gmem:r8.25KB/w12.00MB@logical,r24.75MB/w36.01GB@total,r8.25KB/w12.00MB@cta,r48B/w12.00MB@thread;rmem:r141.15KB/w125.02KB@logical,r423.45MB/w375.07MB@total,r141.15KB/w125.02KB@cta,r776B/w584B@thread footprint=core:528.00KB;gamma_gdn:256B;v39:259:528.00KB;z:528.00KB footprint-precision=exact peak=gmem:36.00MB;rmem:24.02KB persistent=gmem:24.00MB
#   buffer=gamma_gdn holds=1.55MB time=none space=cta.b reuse=32.75KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gated_norm(
    core: Tensor[(1, 1024, 48, 128), "bf16"], z: Tensor[(1, 1024, 6144), "bf16", ((1, 1024, 6144), (17039360, 16640, 1))], gamma_gdn: ConstTensor[(128,), "bf16"], out: Tensor[(1024, 48, 16, 8), "bf16"]
):
    with Mesh((Topology("cta", 3072),), layout=(3072,), names=("b",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(16, 16), names=("row", "lane")) as scope:
            value = T.tensor_view(
                T.ptr_of(out), layout=(1024, 48, 16, 8), shape=(1024, 48, 16, 8)
            )
            T.fill(value, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            value_2 = T.tensor_view(
                T.ptr_of(core), layout=(1, 1024, 48, 128), shape=(1, 1024, 48, 128)
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_2), layout=(1024, 48, 16, 8), shape=(1024, 48, 16, 8)
            )
            value_4 = T.tensor_view(
                T.ptr_of(value_3[cta.b // 3:cta.b // 3 + 1, (cta.b % 3) * 16:(cta.b % 3) * 16 + 1, 0:0 + 1, 0:0 + 1]),
                layout=((1, 16, 16, 8), (0, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            value_5 = T.tensor_view(
                T.ptr_of(value_1),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            dst_view = T.tensor_view(
                T.ptr_of(value_5[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_4, dst_view)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            src_view = T.tensor_view(
                T.ptr_of(value_5[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            T.cast(src_view, x32, dtype="f32")
            value_6 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.row, 1, 1), "rmem"]
            )
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8),
                    "f32",
                    ((1, 16 @ scope.row, 16 @ scope.lane, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.unary(x32, value_8, kind=UnaryKind.SQUARE)
            value_9 = T.tensor_view(
                T.ptr_of(value_7),
                layout=((1, 16 @ scope.row, 16 @ scope.lane, 8), (0, 0, 0, 1)),
                shape=(1, 16, 16, 8),
            )
            dst_view_1 = T.tensor_view(
                T.ptr_of(value_9[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=((1, 16 @ scope.row, 16 @ scope.lane, 8), (0, 0, 0, 1)),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_8, dst_view_1)
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 1, 1),
                    "f32",
                    ((1, 16 @ scope.row, 1, 1, 1), (0, 1, 0, 0, 0)),
                    "rmem",
                ]
            )
            src_view_1 = T.tensor_view(
                T.ptr_of(value_9[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=((1, 16 @ scope.row, 16 @ scope.lane, 8), (0, 0, 0, 1)),
                shape=(1, 16, 16, 8),
            )
            T.reduce(src_view_1, value_10, axes=(2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_11 = T.tensor_view(
                T.ptr_of(value_6), layout=(1, 16 @ scope.row, 1, 1), shape=(1, 16, 1, 1)
            )
            dst_view_2 = T.tensor_view(
                T.ptr_of(value_11[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                layout=(1, 16 @ scope.row, 1, 1),
                shape=(1, 16, 1, 1),
            )
            T.copy(value_10, dst_view_2)
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.row, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            lhs_view = T.tensor_view(
                T.ptr_of(value_11[0:0 + 1, 0:0 + 16, 0:0 + 1, 0:0 + 1]),
                layout=(1, 16 @ scope.row, 1, 1),
                shape=(1, 16, 1, 1),
            )
            T.binary(lhs_view, scalar, value_12, kind=BinaryKind.ADD)
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.row, 1, 1), "rmem"]
            )
            T.unary(value_12, value_13, kind=UnaryKind.RSQRT)
            value_14 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.binary(x32, value_13, value_14, kind=BinaryKind.MUL)
            n = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.cast(value_14, n, dtype="bf16")
            value_15 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.cast(n, value_15, dtype="f32")
            value_16 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.lane, 8), "rmem"]
            )
            value_17 = T.tensor_view(T.ptr_of(gamma_gdn), layout=(128,), shape=(128,))
            value_18 = T.tensor_view(
                T.ptr_of(value_17), layout=((1, 1, 16, 8), (0, 0, 8, 1)), shape=(1, 1, 16, 8)
            )
            value_19 = T.tensor_view(
                T.ptr_of(value_16), layout=(1, 1, 16 @ scope.lane, 8), shape=(1, 1, 16, 8)
            )
            dst_view_3 = T.tensor_view(
                T.ptr_of(value_19[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                layout=(1, 1, 16 @ scope.lane, 8),
                shape=(1, 1, 16, 8),
            )
            T.copy(value_18, dst_view_3)
            value_20 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.lane, 8), "rmem"]
            )
            src_view_2 = T.tensor_view(
                T.ptr_of(value_19[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                layout=(1, 1, 16 @ scope.lane, 8),
                shape=(1, 1, 16, 8),
            )
            T.cast(src_view_2, value_20, dtype="f32")
            value_21 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.binary(value_15, value_20, value_21, kind=BinaryKind.MUL)
            nw = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.cast(value_21, nw, dtype="bf16")
            value_22 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.cast(nw, value_22, dtype="f32")
            value_23 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            value_24 = T.tensor_view(
                T.ptr_of(z),
                layout=((1, 1024, 6144), (17039360, 16640, 1)),
                shape=(1, 1024, 6144),
            )
            value_25 = T.tensor_view(
                T.ptr_of(value_24),
                layout=((1024, 48, 16, 8), (16640, 128, 8, 1)),
                shape=(1024, 48, 16, 8),
            )
            value_26 = T.tensor_view(
                T.ptr_of(value_25[cta.b // 3:cta.b // 3 + 1, (cta.b % 3) * 16:(cta.b % 3) * 16 + 1, 0:0 + 1, 0:0 + 1]),
                layout=((1, 16, 16, 8), (0, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            value_27 = T.tensor_view(
                T.ptr_of(value_23),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            dst_view_4 = T.tensor_view(
                T.ptr_of(value_27[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_26, dst_view_4)
            z32 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            src_view_3 = T.tensor_view(
                T.ptr_of(value_27[0:0 + 1, 0:0 + 16, 0:0 + 16, 0:0 + 8]),
                layout=(1, 16 @ scope.row, 16 @ scope.lane, 8),
                shape=(1, 16, 16, 8),
            )
            T.cast(src_view_3, z32, dtype="f32")
            value_28 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.unary(z32, value_28, kind=UnaryKind.NEG)
            value_29 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.unary(value_28, value_29, kind=UnaryKind.EXP)
            value_30 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_29, scalar_1, value_30, kind=BinaryKind.ADD)
            gate = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.binary(z32, value_30, gate, kind=BinaryKind.DIV)
            value_31 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.binary(value_22, gate, value_31, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.row, 16 @ scope.lane, 8), "rmem"
                ]
            )
            T.cast(value_31, y, dtype="bf16")
            value_32 = T.tensor_view(
                T.ptr_of(out), layout=(1024, 48, 16, 8), shape=(1024, 48, 16, 8)
            )
            window = T.tensor_view(
                T.ptr_of(value_32[cta.b // 3:cta.b // 3 + 1, (cta.b % 3) * 16:(cta.b % 3) * 16 + 16, 0:0 + 16, 0:0 + 8]),
                layout=((1, 16, 16, 8), (6144, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            T.copy(y, window)
