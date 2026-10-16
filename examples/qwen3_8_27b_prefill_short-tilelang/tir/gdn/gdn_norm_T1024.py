# analysis target=nvidia.h200_sxm module=SCHED_GDN_NORM_T1024 function=gdn_norm topology=cta wave=132/6144
# selection requested=memory executed=memory
# memory traffic=gmem:r6.25KB/w12.00MB@logical,r37.50MB/w72.02GB@total,r6.25KB/w12.00MB@cta,r2.05KB/w12.00MB@thread;rmem:r70.96KB/w58.88KB@logical,r425.79MB/w353.30MB@total,r70.96KB/w58.88KB@cta,r716B/w536B@thread footprint=gamma:256B;o:264.00KB;proj:264.00KB;v37:67:2.00KB;v40:68:264.00KB footprint-precision=exact peak=gmem:56.50MB;rmem:12.02KB persistent=gmem:44.50MB
#   buffer=gamma holds=794.25KB time=none space=cta.t,cta.hb reuse=32.75KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_norm(
    o: Tensor[(1024, 48, 128), "bf16"], proj: Tensor[(1024, 16640), "bf16"], gamma: Tensor[(128,), "bf16"], scratch: Tensor[(1024,), "bf16"], out: Tensor[(1024, 48, 128), "bf16"]
):
    with Mesh((Topology("cta", 6144),), layout=(1024, 6), names=("d0", "d1")) as cta:
        with Mesh(
            (Topology("thread", 128),), layout=(4, 2, 16), names=("d0", "d1", "d2")
        ) as scope:
            T.fill(out, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "bf16",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            value_2 = T.tensor_view(
                T.ptr_of(o[cta.d0:cta.d0 + 1, cta.d1 * 8:cta.d1 * 8 + 1, 0:0 + 1]),
                layout=(8, 128),
                shape=(8, 128),
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1),
                layout=((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                shape=(8, 128),
            )
            T.copy(value_2, value_3)
            rf = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_3, rf, dtype="f32")
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(rf, rf, value_4, kind=BinaryKind.MUL)
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 1), "f32", ((4 @ scope.d0, 2 @ scope.d1, 1, 1), (2, 1, 0, 0)), "rmem"
                ]
            )
            T.reduce(value_4, value_5, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
            value_6 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 1), "f32", ((4 @ scope.d0, 2 @ scope.d1, 1, 1), (2, 1, 0, 0)), "rmem"
                ]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 0.0078125)
            T.binary(value_5, scalar, value_6, kind=BinaryKind.MUL)
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 1), "f32", ((4 @ scope.d0, 2 @ scope.d1, 1, 1), (2, 1, 0, 0)), "rmem"
                ]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1e-06)
            T.binary(value_6, scalar_1, value_7, kind=BinaryKind.ADD)
            rr = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 1), "f32", ((4 @ scope.d0, 2 @ scope.d1, 1, 1), (2, 1, 0, 0)), "rmem"
                ]
            )
            T.unary(value_7, rr, kind=UnaryKind.RSQRT)
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(rf, rr, value_8, kind=BinaryKind.MUL)
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "bf16",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_8, value_9, dtype="bf16")
            nb = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_9, nb, dtype="f32")
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[(1, 128), "bf16", ((1, 16 @ scope.d2, 8), (0, 0, 1)), "rmem"]
            )
            value_11 = T.tensor_view(T.ptr_of(gamma), layout=((1, 128), (0, 1)), shape=(1, 128))
            value_12 = T.tensor_view(
                T.ptr_of(value_10), layout=((1, 16 @ scope.d2, 8), (0, 0, 1)), shape=(1, 128)
            )
            T.copy(value_11, value_12)
            gm = T.alloc_tensor(
                tensor_type=Tensor[(1, 128), "f32", ((1, 16 @ scope.d2, 8), (0, 0, 1)), "rmem"]
            )
            T.cast(value_12, gm, dtype="f32")
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(nb, gm, value_13, kind=BinaryKind.MUL)
            value_14 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "bf16",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_13, value_14, dtype="bf16")
            n2 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_14, n2, dtype="f32")
            value_15 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "bf16",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            value_16 = T.tensor_view(
                T.ptr_of(proj[cta.d0:cta.d0 + 1, 10240 + ((cta.d1 * 8) * 128):10240 + ((cta.d1 * 8) * 128) + 1]),
                layout=(8, 128),
                shape=(8, 128),
            )
            value_17 = T.tensor_view(
                T.ptr_of(value_15),
                layout=((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                shape=(8, 128),
            )
            T.copy(value_16, value_17)
            zf = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_17, zf, dtype="f32")
            value_18 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.unary(zf, value_18, kind=UnaryKind.NEG)
            value_19 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.unary(value_18, value_19, kind=UnaryKind.EXP)
            value_20 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_2, 1.0)
            T.binary(value_19, scalar_2, value_20, kind=BinaryKind.ADD)
            value_21 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(zf, value_20, value_21, kind=BinaryKind.DIV)
            value_22 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "f32",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(n2, value_21, value_22, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 128),
                    "bf16",
                    ((4 @ scope.d0, 2 @ scope.d1, 16 @ scope.d2, 8), (0, 0, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_22, y, dtype="bf16")
            value_23 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 1024]), layout=(8, 128), shape=(8, 128)
            )
            value_24 = T.tensor_view(T.ptr_of(value_23), layout=(8, 128), shape=(8, 128))
            T.copy(y, value_24)
            value_25 = T.tensor_view(
                T.ptr_of(value_23), layout=((1, 8, 128), (0, 128, 1)), shape=(1, 8, 128)
            )
            value_26 = T.tensor_view(
                T.ptr_of(out), layout=(1024, 48, 128), shape=(1024, 48, 128)
            )
            window = T.tensor_view(
                T.ptr_of(value_26[cta.d0:cta.d0 + 1, cta.d1 * 8:cta.d1 * 8 + 8, 0:0 + 128]),
                layout=((1, 8, 128), (6144, 128, 1)),
                shape=(1, 8, 128),
            )
            T.copy(value_25, window)
