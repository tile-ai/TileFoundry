# analysis target=nvidia.h200_sxm module=SCHED_ATTN_CORE_T256 function=attn_core topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r368.00KB/w3.06MB@logical,r34.50MB/w294.00MB@total,r368.00KB/w3.06MB@cta,r352.12KB/w3.06MB@thread;rmem:r1.86MB/w1.68MB@logical,r178.33MB/w164.44MB@total,r1.86MB/w1.71MB@cta,r17.11KB/w15.73KB@thread;smem:r480.00KB/w384.00KB@logical,r45.00MB/w36.00MB@total,r480.00KB/w384.00KB@cta,r70.75KB/w320.75KB@thread footprint=diag:16.00KB;q:32.00KB;qkv:32.00KB;v124:112:32.00KB footprint-precision=lower_bound peak=gmem:17.05MB;rmem:160.27KB;smem:160.00KB persistent=gmem:11.02MB
#   buffer=diag holds=112.00KB time=none space=cta.blk,cta.hg reuse=1.48MB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_core(
    q: Tensor[(256, 24, 256), "bf16"], k: Tensor[(256, 4, 256), "bf16"], v: Tensor[(256, 4, 256), "bf16"], qkv: Tensor[(256, 14336), "bf16"], diag: Tensor[(64, 64), "f32"], scratch: Tensor[(16384,), "bf16"], out: Tensor[(256, 6144), "bf16"]
):
    with Mesh((Topology("cta", 96),), layout=(4, 24), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        O0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        m0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        S0_0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        l0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        q_dst0 = T.tensor_view(
            0,
            dtype="bf16",
            storage="smem",
            layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
            shape=(64, 256),
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        k_ring_1_stages = (
            T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        )
        v_ring_1_stages = (
            T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        S0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P0_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l0_2 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        g_dst = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
        )
        value_26 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
        )
        sig = T.alloc_tensor(tensor_type=Tensor[(64, 256), "bf16", ((64, 256), (0, 1)), "rmem"])
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "f32", ((64, 256), (0, 1)), "rmem"]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "f32", ((64, 256), (0, 1)), "rmem"]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "f32", ((64, 256), (0, 1)), "rmem"]
        )
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "f32", ((64, 256), (0, 1)), "rmem"]
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "f32", ((64, 256), (0, 1)), "rmem"]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[(64, 256), "bf16", ((64, 256), (0, 1)), "rmem"]
        )
        value_33 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
        )
        y0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_34 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_35 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        ob0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_36 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_37 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_38 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_39 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_40_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
        )
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_1:
                with scope_1 as threads:
                    T.fill(O0, 0.0)
                    T.fill(S0_0, 0.0)
                    src_frame = T.tensor_view(
                        T.ptr_of(S0_0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads.d1, 2, 4 @ threads.d0, 2, 4 @ threads.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads.d1, 2, 4 @ threads.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame,
                        dst_frame,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, -1e999)
                with scope_1 as threads_3:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame, scalar, dst_frame_1, kind=BinaryKind.ADD)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(S0_0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(l0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_1,
                        dst_frame_2,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(diag, dst_frame_3)
            with scope[:1, :32] as scope_2:
                tile = T.tensor_view(
                    T.ptr_of(q[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, cta.d1:cta.d1 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (6144, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_2, layout=(32,), names=("d0",)) as threads_6:
                    T.copy_async_tensor(tile_1, q_dst0)
            k_ring_stages = (
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            v_ring_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for j in range(0, (0 - cta.d0) + 3, 1):
                with scope[:1, :32] as scope_3:
                    tile_2 = T.tensor_view(
                        T.ptr_of(k[j * 64:j * 64 + 64, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 256]),
                        layout=((64, 1, 256), (1024, 256, 1)),
                        shape=(64, 1, 256),
                    )
                    tile_3 = T.tensor_view(
                        T.ptr_of(tile_2[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_7:
                        T.copy_async_tensor(tile_3, k_ring_stages[j % 2])
                with scope[:1, 32:64] as scope_4:
                    tile_4 = T.tensor_view(
                        T.ptr_of(v[j * 64:j * 64 + 64, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 256]),
                        layout=((64, 1, 256), (1024, 256, 1)),
                        shape=(64, 1, 256),
                    )
                    tile_5 = T.tensor_view(
                        T.ptr_of(tile_4[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_8:
                        T.copy_async_tensor(tile_5, v_ring_stages[j % 2])
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    with scope_5 as threads_9:
                        T.fill(value_6, 0.0)
                    tile_6 = T.tensor_view(
                        T.ptr_of(k_ring_stages[j % 2][0:0 + 64, 0:0 + 256]),
                        layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(256, 64),
                    )
                    with scope_5 as threads_10:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 64, 64):
                                for o_k in range(0, 256, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(value_6[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_6[o_k:o_k + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_10),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(value_6[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_10),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(value_6[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_10),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(value_6[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_10),
                                    )
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_2,
                            dst_frame_4,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.MAX,
                        )
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_1, rhs_frame, dst_frame_5, kind=BinaryKind.MAX)
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_6 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_2, rhs_frame_1, dst_frame_6, kind=BinaryKind.SUB)
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 0.09016844005556021)
                    with scope_5 as threads_14:
                        lhs_frame_3 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_7 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_3, scalar_1, dst_frame_7, kind=BinaryKind.MUL)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_8 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.unary(src_frame_3, dst_frame_8, kind=UnaryKind.EXP2)
                        lhs_frame_4 = T.tensor_view(
                            T.ptr_of(O0[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_9 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        T.binary(lhs_frame_4, rhs_frame_2, dst_frame_9, kind=BinaryKind.MUL)
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    with scope_5 as threads_17:
                        lhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_5, scalar_2, dst_frame_10, kind=BinaryKind.MUL)
                    scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_3, 0.09016844005556021)
                    with scope_5 as threads_18:
                        lhs_frame_6 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_6, scalar_3, dst_frame_11, kind=BinaryKind.MUL)
                        lhs_frame_7 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_7, rhs_frame_3, dst_frame_12, kind=BinaryKind.SUB)
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.unary(src_frame_4, dst_frame_13, kind=UnaryKind.EXP2)
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_5, dst_frame_14, dtype="bf16")
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_6, dst_frame_15)
                        for o_m_1 in range(0, 64, 64):
                            for o_n_1 in range(0, 256, 256):
                                for o_k_1 in range(0, 64, 16):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(value_2[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                        layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_7[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(v_ring_stages[j % 2][o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_18),
                                    )
                        lhs_frame_8 = T.tensor_view(
                            T.ptr_of(l0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_4 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_8, rhs_frame_4, dst_frame_16, kind=BinaryKind.MUL)
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 2, 4 @ threads_18.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_17 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_7,
                            dst_frame_17,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.SUM,
                        )
                        lhs_frame_9 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_18 = T.tensor_view(
                            T.ptr_of(l0_1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_18.d1, 2, 4 @ threads_18.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_9, rhs_frame_5, dst_frame_18, kind=BinaryKind.ADD)
                    l0_1_view = T.tensor_view(
                        T.ptr_of(l0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ scope_5.d1, 2, 4 @ scope_5.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(l0_1_view, l0)
                    value_2_view = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ scope_5.d1, 2, 4 @ scope_5.d0, 2, 4 @ scope_5.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(value_2_view, O0)
                    m_new0_view = T.tensor_view(
                        T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ scope_5.d1, 2, 4 @ scope_5.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(m_new0_view, m0)
            with scope[:1, :32] as scope_6:
                tile_7 = T.tensor_view(
                    T.ptr_of(k[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (1024, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_8 = T.tensor_view(
                    T.ptr_of(tile_7[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_30:
                    T.copy_async_tensor(tile_8, k_ring_1_stages[0])
            with scope[:1, 32:64] as scope_7:
                tile_9 = T.tensor_view(
                    T.ptr_of(v[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (1024, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_10 = T.tensor_view(
                    T.ptr_of(tile_9[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_7, layout=(32,), names=("d0",)) as threads_31:
                    T.copy_async_tensor(tile_10, v_ring_1_stages[0])
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_8:
                with scope_8 as threads_32:
                    T.fill(value_18, 0.0)
                tile_11 = T.tensor_view(
                    T.ptr_of(k_ring_1_stages[0][0:0 + 64, 0:0 + 256]),
                    layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                    shape=(256, 64),
                )
                with scope_8 as threads_33:
                    for o_m_2 in range(0, 64, 64):
                        for o_n_2 in range(0, 64, 64):
                            for o_k_2 in range(0, 256, 64):
                                acc_view_5 = T.tensor_view(
                                    T.ptr_of(value_18[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                    layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_5 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_5 = T.tensor_view(
                                    T.ptr_of(tile_11[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_5,
                                    lhs_view_5,
                                    rhs_view_5,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                )
                                acc_view_6 = T.tensor_view(
                                    T.ptr_of(value_18[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                    layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_6 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_6 = T.tensor_view(
                                    T.ptr_of(tile_11[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_6,
                                    lhs_view_6,
                                    rhs_view_6,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                )
                                acc_view_7 = T.tensor_view(
                                    T.ptr_of(value_18[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                    layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_7 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_7 = T.tensor_view(
                                    T.ptr_of(tile_11[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_7,
                                    lhs_view_7,
                                    rhs_view_7,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                )
                                acc_view_8 = T.tensor_view(
                                    T.ptr_of(value_18[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                    layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_8 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_8 = T.tensor_view(
                                    T.ptr_of(tile_11[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_8,
                                    lhs_view_8,
                                    rhs_view_8,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                )
                    lhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_19 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_10, rhs_frame_6, dst_frame_19, kind=BinaryKind.ADD)
                    src_frame_8 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_20 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_8,
                        dst_frame_20,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                    lhs_frame_11 = T.tensor_view(
                        T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_7 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_21 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_11, rhs_frame_7, dst_frame_21, kind=BinaryKind.MAX)
                    lhs_frame_12 = T.tensor_view(
                        T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_8 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_22 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_12, rhs_frame_8, dst_frame_22, kind=BinaryKind.SUB)
                scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_4, 0.09016844005556021)
                with scope_8 as threads_38:
                    lhs_frame_13 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_23 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_13, scalar_4, dst_frame_23, kind=BinaryKind.MUL)
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_24 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.unary(src_frame_9, dst_frame_24, kind=UnaryKind.EXP2)
                    lhs_frame_14 = T.tensor_view(
                        T.ptr_of(O0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_9 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_25 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_14, rhs_frame_9, dst_frame_25, kind=BinaryKind.MUL)
                scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_5, 0.09016844005556021)
                with scope_8 as threads_41:
                    lhs_frame_15 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_41.d1, 2, 4 @ threads_41.d0, 2, 4 @ threads_41.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_26 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_41.d1, 2, 4 @ threads_41.d0, 2, 4 @ threads_41.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_15, scalar_5, dst_frame_26, kind=BinaryKind.MUL)
                scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_6, 0.09016844005556021)
                with scope_8 as threads_42:
                    lhs_frame_16 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_27 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_16, scalar_6, dst_frame_27, kind=BinaryKind.MUL)
                    lhs_frame_17 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_28 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_17, rhs_frame_10, dst_frame_28, kind=BinaryKind.SUB)
                    src_frame_10 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_29 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.unary(src_frame_10, dst_frame_29, kind=UnaryKind.EXP2)
                    src_frame_11 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_30 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.cast(src_frame_11, dst_frame_30, dtype="bf16")
                    src_frame_12 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_31 = T.tensor_view(
                        T.ptr_of(value_19[0:0 + 64, 0:0 + 64]),
                        layout=((4, 8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(src_frame_12, dst_frame_31)
                    for o_m_3 in range(0, 64, 64):
                        for o_n_3 in range(0, 256, 256):
                            for o_k_3 in range(0, 64, 16):
                                acc_view_9 = T.tensor_view(
                                    T.ptr_of(value_14[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                    layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 32), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 256),
                                )
                                lhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_19[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                    layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 2), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 16),
                                )
                                rhs_view_9 = T.tensor_view(
                                    T.ptr_of(v_ring_1_stages[0][o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_9,
                                    lhs_view_9,
                                    rhs_view_9,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_42),
                                )
                    lhs_frame_18 = T.tensor_view(
                        T.ptr_of(l0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_11 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_32 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_18, rhs_frame_11, dst_frame_32, kind=BinaryKind.MUL)
                    src_frame_13 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_33 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_13,
                        dst_frame_33,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    lhs_frame_19 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_12 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_34 = T.tensor_view(
                        T.ptr_of(l0_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_19, rhs_frame_12, dst_frame_34, kind=BinaryKind.ADD)
            with scope[:1, 32:64] as scope_9:
                tile_12 = T.tensor_view(
                    T.ptr_of(qkv[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, (cta.d1 * 512) + 256:(cta.d1 * 512) + 256 + 256]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_9, layout=(32,), names=("d0",)) as threads_51:
                    T.copy_async_tensor(tile_12, g_dst)
            with Mesh(scope[:1, 64:], layout=(64,), names=("d0",)) as scope_10:
                scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_7, 1.0)
                with scope_10 as threads_52:
                    dst_frame_35 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_52.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.copy(g_dst, dst_frame_35)
                    src_frame_14 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_52.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_36 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_52.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_14, dst_frame_36, dtype="f32")
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, -1.4426950408889634)
                with scope_10 as threads_54:
                    lhs_frame_20 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_54.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_37 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_54.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_20, scalar_8, dst_frame_37, kind=BinaryKind.MUL)
                    src_frame_15 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_54.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_38 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_54.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.unary(src_frame_15, dst_frame_38, kind=UnaryKind.EXP2)
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 1.0)
                with scope_10 as threads_56:
                    lhs_frame_21 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_39 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_21, scalar_9, dst_frame_39, kind=BinaryKind.ADD)
                    rhs_frame_13 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_40 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.binary(scalar_7, rhs_frame_13, dst_frame_40, kind=BinaryKind.DIV)
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    dst_frame_41 = T.tensor_view(
                        T.ptr_of(sig[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_16, dst_frame_41, dtype="bf16")
                    src_frame_17 = T.tensor_view(
                        T.ptr_of(sig[0:0 + 64, 0:0 + 256]),
                        layout=((64 @ threads_56.d0, 256), (0, 1)),
                        shape=(64, 256),
                    )
                    T.copy(src_frame_17, value_26)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_11:
                scalar_10 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_10, 1.0)
                with scope_11 as threads_60:
                    rhs_frame_14 = T.tensor_view(
                        T.ptr_of(l0_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_42 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(scalar_10, rhs_frame_14, dst_frame_42, kind=BinaryKind.DIV)
                    lhs_frame_22 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_15 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_43 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_22, rhs_frame_15, dst_frame_43, kind=BinaryKind.MUL)
                    src_frame_18 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_44 = T.tensor_view(
                        T.ptr_of(ob0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_18, dst_frame_44, dtype="bf16")
                    src_frame_19 = T.tensor_view(
                        T.ptr_of(ob0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_45 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_19, dst_frame_45, dtype="f32")
                    dst_frame_46 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(value_26, dst_frame_46)
                    src_frame_20 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_47 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_20, dst_frame_47, dtype="f32")
                    lhs_frame_23 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_16 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_48 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_23, rhs_frame_16, dst_frame_48, kind=BinaryKind.MUL)
                    src_frame_21 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_49 = T.tensor_view(
                        T.ptr_of(y0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_21, dst_frame_49, dtype="bf16")
                    src_frame_22 = T.tensor_view(
                        T.ptr_of(y0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_60.d1, 2, 4 @ threads_60.d0, 2, 4 @ threads_60.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(src_frame_22, value_33)
            with scope[:1, :32] as scope_12:
                with Mesh(scope_12, layout=(32,), names=("d0",)) as threads_69:
                    T.copy_async_tensor(value_33, value_40_1)
                window = T.tensor_view(
                    T.ptr_of(out[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, cta.d1 * 256:cta.d1 * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_40_1, window)
