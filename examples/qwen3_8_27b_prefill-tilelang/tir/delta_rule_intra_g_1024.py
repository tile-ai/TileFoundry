# analysis target=nvidia.h200_sxm module=DeltaRuleIntra function=intra_g topology=cta wave=132/768
# selection requested=memory executed=memory
# memory traffic=gmem:r768B/w192.25KB@logical,r576.00KB/w144.19MB@total,r768B/w192.25KB@cta,r264B/w192.00KB@thread;rmem:r50.05KB/w50.01KB@logical,r37.54MB/w37.51MB@total,r50.05KB/w50.01KB@cta,r1.33KB/w1.28KB@thread;smem:r256B/w256B@logical,r192.00KB/w192.00KB@total,r256B/w256B@cta,r256B/w4B@thread footprint=g:256B;iota:256B;v21:70:256B footprint-precision=exact peak=gmem:576.25KB;rmem:16.77KB;smem:256B persistent=gmem:192.25KB
#   buffer=iota holds=768B time=none space=cta.c,cta.h reuse=32.75KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def intra_g(
    g: Tensor[(1024, 48), "f32"], iota: Tensor[(64,), "i32"], out: Tensor[(48, 1024, 1), "f32"]
):
    with Mesh((Topology("cta", 768),), layout=(16, 48), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        gcum = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", (64, 1), "rmem"])
        value = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", ((64, 1), (1, 0)), "rmem"])
        gm = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", ((64, 64), (0, 1)), "rmem"])
        value_1 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", (64, 64), "rmem"])
        gv = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", (1, 64), "rmem"])
        gsm = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
        g1 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", (64, 64), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "bool", (64, 64), "rmem"])
        jc = T.alloc_tensor(tensor_type=Tensor[(1, 64), "i32", (1, 64), "rmem"])
        ir = T.alloc_tensor(tensor_type=Tensor[(64, 1), "i32", (64, 1), "rmem"])
        with Mesh((Topology("thread", 128),), layout=(2, 64), names=("d0", "d1")) as scope:
            with Mesh(scope[:1], layout=(64,), names=("d0",)) as scope_1:
                tile = T.tensor_view(
                    T.ptr_of(g[cta.d0 * 64:cta.d0 * 64 + 64, cta.d1:cta.d1 + 1]),
                    layout=((64, 1), (48, 1)),
                    shape=(64, 1),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[0:0 + 64, 0:0 + 1]), layout=((64,), (48,)), shape=(64,)
                )
                with scope_1 as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(g1[0:0 + 64]), layout=(64 @ threads.d0,), shape=(64,)
                    )
                    T.copy(tile_1, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(g1[0:0 + 64]), layout=(64 @ threads.d0,), shape=(64,)
                    )
                    T.copy(src_frame, gsm)
                tile_2 = T.tensor_view(T.ptr_of(gsm[0:0 + 64]), layout=(1, 64), shape=(1, 64))
                with scope_1 as threads_2:
                    T.copy(tile_2, gv)
                tile_3 = T.tensor_view(T.ptr_of(iota[0:0 + 64]), layout=(1, 64), shape=(1, 64))
                with scope_1 as threads_3:
                    T.copy(tile_3, jc)
                tile_4 = T.tensor_view(T.ptr_of(iota[0:0 + 64]), layout=(64, 1), shape=(64, 1))
                with scope_1 as threads_4:
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(ir[0:0 + 64, 0:0 + 1]),
                        layout=(64 @ threads_4.d0, 1),
                        shape=(64, 1),
                    )
                    T.copy(tile_4, dst_frame_1)
                    rhs_frame = T.tensor_view(
                        T.ptr_of(ir[0:0 + 64, 0:0 + 1]),
                        layout=(64 @ threads_4.d0, 1),
                        shape=(64, 1),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    T.binary(jc, rhs_frame, dst_frame_2, kind=BinaryKind.LE)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    T.cast(src_frame_1, dst_frame_3, dtype="f32")
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    T.binary(gv, rhs_frame_1, dst_frame_4, kind=BinaryKind.MUL)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (64, 1)),
                        shape=(64, 64),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(gm[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (0, 1)),
                        shape=(64, 64),
                    )
                    T.copy(src_frame_2, dst_frame_5)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(gm[0:0 + 64, 0:0 + 64]),
                        layout=((64 @ threads_4.d0, 64), (0, 1)),
                        shape=(64, 64),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((64 @ threads_4.d0, 1), (1, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_3,
                        dst_frame_6,
                        axes=(1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((64 @ threads_4.d0, 1), (1, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(gcum[0:0 + 64, 0:0 + 1]),
                        layout=(64 @ threads_4.d0, 1),
                        shape=(64, 1),
                    )
                    T.copy(src_frame_4, dst_frame_7)
                tile_5 = T.tensor_view(
                    T.ptr_of(gcum[0:0 + 64, 0:0 + 1]), layout=(1, 64, 1), shape=(1, 64, 1)
                )
                window = T.tensor_view(
                    T.ptr_of(out[cta.d1:cta.d1 + 1, cta.d0 * 64:cta.d0 * 64 + 64, 0:0 + 1]),
                    layout=((1, 64, 1), (1024, 1, 1)),
                    shape=(1, 64, 1),
                )
                T.copy(tile_5, window)
