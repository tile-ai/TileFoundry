# analysis target=nvidia.h200_sxm module=Prep function=gate_beta topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w102.00KB@logical,r7.50MB/w127.50MB@total,r6.00KB/w102.00KB@cta,r48B/w96.05KB@thread;rmem:r72.10KB/w72.01KB@logical,r90.13MB/w90.01MB@total,r72.10KB/w72.01KB@cta,r680B/w584B@thread footprint=b:12.00KB;v15:192:12.00KB footprint-precision=exact peak=gmem:192.00KB;rmem:18.01KB persistent=gmem:96.00KB
#   buffer=b holds=24.00KB time=none space=cta.cb reuse=948.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gate_beta(
    b: Tensor[(1, 1024, 48), "bf16", ((1, 1024, 48), (17039360, 16640, 1))], out: Tensor[(128, 8, 16, 3), "bf16"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("bt", "cb")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("s", "g")) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3))
            T.fill(value, 0.0)
            for _ in range(cta.cb, 1, 80):
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                value_2 = T.tensor_view(
                    T.ptr_of(b),
                    layout=((1, 1024, 48), (17039360, 16640, 1)),
                    shape=(1, 1024, 48),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_2),
                    layout=((128, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(128, 8, 16, 3),
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=((128, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(128, 8, 16, 3),
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_4[cta.bt * (64 // 8):cta.bt * (64 // 8) + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((8, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(8, 8, 16, 3),
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_5, dst_view)
                bb = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                T.cast(src_view, bb, dtype="f32")
                value_7 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.unary(bb, value_7, kind=UnaryKind.NEG)
                value_8 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.unary(value_7, value_8, kind=UnaryKind.EXP)
                value_9 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(value_8, scalar, value_9, kind=BinaryKind.ADD)
                value_10 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(scalar_1, value_9, value_10, kind=BinaryKind.DIV)
                value_11 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.cast(value_10, value_11, dtype="bf16")
                value_12 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                value_13 = T.tensor_view(
                    T.ptr_of(value_12), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                value_14 = T.tensor_view(
                    T.ptr_of(value_13), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                window = T.tensor_view(
                    T.ptr_of(value_14[cta.bt * (64 // 8):cta.bt * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_11, window)
