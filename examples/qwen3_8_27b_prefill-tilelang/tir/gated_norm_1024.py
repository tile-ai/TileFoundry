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
    with Mesh((Topology("cta", 3072),), layout=(3072,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(16, 16), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            value_1 = T.tensor_view(
                T.ptr_of(core[0:0 + 1, cta.d0 // 3:cta.d0 // 3 + 1, (cta.d0 % 3) * 16:(cta.d0 % 3) * 16 + 1, 0:0 + 1]),
                layout=((1, 16, 16, 8), (0, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            value_2 = T.tensor_view(
                T.ptr_of(value),
                layout=(1, 16 @ scope.d0, 16 @ scope.d1, 8),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_1, value_2)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(value_2, x32, dtype="f32")
            value_3 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.d0, 1, 1), "rmem"]
            )
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8),
                    "f32",
                    ((1, 16 @ scope.d0, 16 @ scope.d1, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.unary(x32, value_5, kind=UnaryKind.SQUARE)
            value_6 = T.tensor_view(
                T.ptr_of(value_4),
                layout=((1, 16 @ scope.d0, 16 @ scope.d1, 8), (0, 0, 0, 1)),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_5, value_6)
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 1, 1), "f32", ((1, 16 @ scope.d0, 1, 1, 1), (0, 1, 0, 0, 0)), "rmem"
                ]
            )
            T.reduce(value_6, value_7, axes=(2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_8 = T.tensor_view(
                T.ptr_of(value_3), layout=(1, 16 @ scope.d0, 1, 1), shape=(1, 16, 1, 1)
            )
            T.copy(value_7, value_8)
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.d0, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(value_8, scalar, value_9, kind=BinaryKind.ADD)
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[(1, 16, 1, 1), "f32", (1, 16 @ scope.d0, 1, 1), "rmem"]
            )
            T.unary(value_9, value_10, kind=UnaryKind.RSQRT)
            value_11 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.binary(x32, value_10, value_11, kind=BinaryKind.MUL)
            n = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(value_11, n, dtype="bf16")
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(n, value_12, dtype="f32")
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.d1, 8), "rmem"]
            )
            value_14 = T.tensor_view(
                T.ptr_of(gamma_gdn), layout=((1, 1, 16, 8), (0, 0, 8, 1)), shape=(1, 1, 16, 8)
            )
            value_15 = T.tensor_view(
                T.ptr_of(value_13), layout=(1, 1, 16 @ scope.d1, 8), shape=(1, 1, 16, 8)
            )
            T.copy(value_14, value_15)
            value_16 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.d1, 8), "rmem"]
            )
            T.cast(value_15, value_16, dtype="f32")
            value_17 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.binary(value_12, value_16, value_17, kind=BinaryKind.MUL)
            nw = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(value_17, nw, dtype="bf16")
            value_18 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(nw, value_18, dtype="f32")
            value_19 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            value_20 = T.tensor_view(
                T.ptr_of(z[0:0 + 1, cta.d0 // 3:cta.d0 // 3 + 1, 128 * ((cta.d0 % 3) * 16):128 * ((cta.d0 % 3) * 16) + 1]),
                layout=((1, 16, 16, 8), (0, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            value_21 = T.tensor_view(
                T.ptr_of(value_19),
                layout=(1, 16 @ scope.d0, 16 @ scope.d1, 8),
                shape=(1, 16, 16, 8),
            )
            T.copy(value_20, value_21)
            z32 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(value_21, z32, dtype="f32")
            value_22 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.unary(z32, value_22, kind=UnaryKind.NEG)
            value_23 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.unary(value_22, value_23, kind=UnaryKind.EXP)
            value_24 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_23, scalar_1, value_24, kind=BinaryKind.ADD)
            gate = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.binary(z32, value_24, gate, kind=BinaryKind.DIV)
            value_25 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "f32", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.binary(value_18, gate, value_25, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16, 8), "bf16", (1, 16 @ scope.d0, 16 @ scope.d1, 8), "rmem"
                ]
            )
            T.cast(value_25, y, dtype="bf16")
            value_26 = T.tensor_view(
                T.ptr_of(out), layout=(1024, 48, 16, 8), shape=(1024, 48, 16, 8)
            )
            window = T.tensor_view(
                T.ptr_of(value_26[cta.d0 // 3:cta.d0 // 3 + 1, (cta.d0 % 3) * 16:(cta.d0 % 3) * 16 + 16, 0:0 + 16, 0:0 + 8]),
                layout=((1, 16, 16, 8), (6144, 128, 8, 1)),
                shape=(1, 16, 16, 8),
            )
            T.copy(y, window)
