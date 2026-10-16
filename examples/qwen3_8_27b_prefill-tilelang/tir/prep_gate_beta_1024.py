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
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            for _ in range(cta.d1, 1, 80):
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(b[0:0 + 1, 8 * (cta.d0 * (64 // 8)):8 * (cta.d0 * (64 // 8)) + 1, 0:0 + 1]),
                    layout=((8, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(8, 8, 16, 3),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_1, value_2)
                bb = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.cast(value_2, bb, dtype="f32")
                value_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.unary(bb, value_3, kind=UnaryKind.NEG)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.unary(value_3, value_4, kind=UnaryKind.EXP)
                value_5 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(value_4, scalar, value_5, kind=BinaryKind.ADD)
                value_6 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(scalar_1, value_5, value_6, kind=BinaryKind.DIV)
                value_7 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.cast(value_6, value_7, dtype="bf16")
                value_8 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                window = T.tensor_view(
                    T.ptr_of(value_8[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_7, window)
