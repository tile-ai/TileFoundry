# analysis target=nvidia.h200_sxm module=SCHED_ATTN_CORE_T1024 function=attn_core topology=cta wave=132/192
# selection requested=memory executed=memory
# memory traffic=gmem:r1.22MB/w12.12MB@logical,r234.00MB/w2.27GB@total,r1.22MB/w12.12MB@cta,r1.19MB/w12.12MB@thread;rmem:r9.73MB/w8.23MB@logical,r1.82GB/w1.62GB@total,r9.73MB/w8.67MB@cta,r70.12KB/w61.27KB@thread;smem:r3.12MB/w1.19MB@logical,r600.00MB/w228.00MB@total,r3.12MB/w1.19MB@cta,r368.50KB/w1.13MB@thread footprint=diag:16.00KB;v222:142:32.00KB;v229:144:32.00KB footprint-precision=lower_bound peak=gmem:68.05MB;rmem:240.52KB;smem:224.00KB persistent=gmem:44.02MB
#   buffer=diag holds=80.00KB time=none space=cta.blk,cta.hg reuse=2.05MB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_core(
    q: Tensor[(1024, 24, 256), "bf16"], k: Tensor[(1024, 4, 256), "bf16"], v: Tensor[(1024, 4, 256), "bf16"], qkv: Tensor[(1024, 14336), "bf16"], diag: Tensor[(64, 64), "f32"], scratch: Tensor[(16384,), "bf16"], scratch_1: Tensor[(16384,), "bf16"], out: Tensor[(1024, 6144), "bf16"]
):
    with Mesh((Topology("cta", 192),), layout=(16, 12), names=("d0", "d1")) as cta:
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
        O1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        m1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        S0_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        l1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_3 = T.alloc_tensor(
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
        q_dst1 = T.tensor_view(
            32768,
            dtype="bf16",
            storage="smem",
            layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
            shape=(64, 256),
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l1_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new0 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        k_ring_1_stages = (
            T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        )
        v_ring_1_stages = (
            T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new0_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        S0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_33 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_34 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P0_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_35 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_36 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_37 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l0_2 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_38 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_39 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        g_dst0 = T.tensor_view(
            0,
            dtype="bf16",
            storage="smem",
            layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
            shape=(64, 256),
        )
        g_dst1 = T.tensor_view(
            196608,
            dtype="bf16",
            storage="smem",
            layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
            shape=(64, 256),
        )
        value_40 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
        )
        y0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_41 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_42 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        ob0 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_43 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_44 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_45 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_46 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_47 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_48 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_49 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_50 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_51 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_52 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_53 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        alpha1_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_54 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_55 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        m_new1_1 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_56 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        S1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_57 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_58 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_59 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        P1_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_60 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_61 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_62 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        l1_2 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_63 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_64 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_65 = T.tensor_view(
            32768, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
        )
        y1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_66 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_67 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        ob1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_68 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_69 = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"]
        )
        value_70 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_71 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_72 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_73 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_74 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_75 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_76 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "f32", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_77 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 256), "bf16", ((8, 2, 4, 2, 4, 32), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_78_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
        )
        value_79_1 = T.tensor_view(
            T.ptr_of(scratch_1[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_1:
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
            with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_2:
                with scope_2 as threads_6:
                    T.fill(O1, 0.0)
                    T.fill(S0_1, 0.0)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(S0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_2,
                        dst_frame_4,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, -1e999)
                with scope_2 as threads_9:
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(m1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_1, scalar_1, dst_frame_5, kind=BinaryKind.ADD)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(S0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(l1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_3,
                        dst_frame_6,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(diag, dst_frame_7)
            with scope[:1, :32] as scope_3:
                tile = T.tensor_view(
                    T.ptr_of(q[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, (cta.d1 * 2) + 0:(cta.d1 * 2) + 0 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (6144, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_12:
                    T.copy_async_tensor(tile_1, q_dst0)
                tile_2 = T.tensor_view(
                    T.ptr_of(q[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, (cta.d1 * 2) + 1:(cta.d1 * 2) + 1 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (6144, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_3 = T.tensor_view(
                    T.ptr_of(tile_2[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_13:
                    T.copy_async_tensor(tile_3, q_dst1)
            k_ring_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            v_ring_stages = (
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for j in range(0, (0 - cta.d0) + 15, 1):
                with scope[:1, :32] as scope_4:
                    tile_4 = T.tensor_view(
                        T.ptr_of(k[j * 64:j * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 256]),
                        layout=((64, 1, 256), (1024, 256, 1)),
                        shape=(64, 1, 256),
                    )
                    tile_5 = T.tensor_view(
                        T.ptr_of(tile_4[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_14:
                        T.copy_async_tensor(tile_5, k_ring_stages[j % 2])
                with scope[:1, 32:64] as scope_5:
                    tile_6 = T.tensor_view(
                        T.ptr_of(v[j * 64:j * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 256]),
                        layout=((64, 1, 256), (1024, 256, 1)),
                        shape=(64, 1, 256),
                    )
                    tile_7 = T.tensor_view(
                        T.ptr_of(tile_6[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    with Mesh(scope_5, layout=(32,), names=("d0",)) as threads_15:
                        T.copy_async_tensor(tile_7, v_ring_stages[j % 2])
                with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_6:
                    with scope_6 as threads_16:
                        T.fill(value_8, 0.0)
                    tile_8 = T.tensor_view(
                        T.ptr_of(k_ring_stages[j % 2][0:0 + 64, 0:0 + 256]),
                        layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(256, 64),
                    )
                    with scope_6 as threads_17:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 64, 64):
                                for o_k in range(0, 256, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(value_8[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(q_dst1[o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_8[o_k:o_k + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_17),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(value_8[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(q_dst1[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_17),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(value_8[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(q_dst1[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_17),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(value_8[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(q_dst1[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_17),
                                    )
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_8 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_4,
                            dst_frame_8,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.MAX,
                        )
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(m1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_9 = T.tensor_view(
                            T.ptr_of(m_new1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_2, rhs_frame, dst_frame_9, kind=BinaryKind.MAX)
                        lhs_frame_3 = T.tensor_view(
                            T.ptr_of(m1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(m_new1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_17.d1, 2, 4 @ threads_17.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_3, rhs_frame_1, dst_frame_10, kind=BinaryKind.SUB)
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    with scope_6 as threads_21:
                        lhs_frame_4 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_4, scalar_2, dst_frame_11, kind=BinaryKind.MUL)
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(alpha1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.unary(src_frame_5, dst_frame_12, kind=UnaryKind.EXP2)
                        lhs_frame_5 = T.tensor_view(
                            T.ptr_of(O1[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(alpha1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        T.binary(lhs_frame_5, rhs_frame_2, dst_frame_13, kind=BinaryKind.MUL)
                    scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_3, 0.09016844005556021)
                    with scope_6 as threads_24:
                        lhs_frame_6 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_24.d1, 2, 4 @ threads_24.d0, 2, 4 @ threads_24.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_24.d1, 2, 4 @ threads_24.d0, 2, 4 @ threads_24.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_6, scalar_3, dst_frame_14, kind=BinaryKind.MUL)
                    scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_4, 0.09016844005556021)
                    with scope_6 as threads_25:
                        lhs_frame_7 = T.tensor_view(
                            T.ptr_of(m_new1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_7, scalar_4, dst_frame_15, kind=BinaryKind.MUL)
                        lhs_frame_8 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_8, rhs_frame_3, dst_frame_16, kind=BinaryKind.SUB)
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_17 = T.tensor_view(
                            T.ptr_of(P1[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.unary(src_frame_6, dst_frame_17, kind=UnaryKind.EXP2)
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(P1[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_18 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_7, dst_frame_18, dtype="bf16")
                        src_frame_8 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_19 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_8, dst_frame_19)
                        for o_m_1 in range(0, 64, 64):
                            for o_n_1 in range(0, 256, 256):
                                for o_k_1 in range(0, 64, 16):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(value_4[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                        layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_9[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 2), (1, 8, 16, 64, 128, 512)),
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
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_25),
                                    )
                        lhs_frame_9 = T.tensor_view(
                            T.ptr_of(l1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_4 = T.tensor_view(
                            T.ptr_of(alpha1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_20 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_9, rhs_frame_4, dst_frame_20, kind=BinaryKind.MUL)
                        src_frame_9 = T.tensor_view(
                            T.ptr_of(P1[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_21 = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_9,
                            dst_frame_21,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.SUM,
                        )
                        lhs_frame_10 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_22 = T.tensor_view(
                            T.ptr_of(l1_1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_10, rhs_frame_5, dst_frame_22, kind=BinaryKind.ADD)
                with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_7:
                    with scope_7 as threads_34:
                        T.fill(value_20, 0.0)
                    tile_9 = T.tensor_view(
                        T.ptr_of(k_ring_stages[j % 2][0:0 + 64, 0:0 + 256]),
                        layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(256, 64),
                    )
                    with scope_7 as threads_35:
                        for o_m_2 in range(0, 64, 64):
                            for o_n_2 in range(0, 64, 64):
                                for o_k_2 in range(0, 256, 64):
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(value_20[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 2, 4 @ threads_35.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_9[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_35),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(value_20[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 2, 4 @ threads_35.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_9[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_35),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(value_20[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 2, 4 @ threads_35.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_9[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_35),
                                    )
                                    acc_view_8 = T.tensor_view(
                                        T.ptr_of(value_20[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 2, 4 @ threads_35.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_8 = T.tensor_view(
                                        T.ptr_of(q_dst0[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_8 = T.tensor_view(
                                        T.ptr_of(tile_9[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_8,
                                        lhs_view_8,
                                        rhs_view_8,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_35),
                                    )
                        src_frame_10 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 2, 4 @ threads_35.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_23 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_10,
                            dst_frame_23,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.MAX,
                        )
                        lhs_frame_11 = T.tensor_view(
                            T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_6 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_24 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_11, rhs_frame_6, dst_frame_24, kind=BinaryKind.MAX)
                        lhs_frame_12 = T.tensor_view(
                            T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_7 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_25 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_35.d1, 2, 4 @ threads_35.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_12, rhs_frame_7, dst_frame_25, kind=BinaryKind.SUB)
                    scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_5, 0.09016844005556021)
                    with scope_7 as threads_39:
                        lhs_frame_13 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_26 = T.tensor_view(
                            T.ptr_of(value_17[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_13, scalar_5, dst_frame_26, kind=BinaryKind.MUL)
                        src_frame_11 = T.tensor_view(
                            T.ptr_of(value_17[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_27 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.unary(src_frame_11, dst_frame_27, kind=UnaryKind.EXP2)
                        lhs_frame_14 = T.tensor_view(
                            T.ptr_of(O0[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 2, 4 @ threads_39.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        rhs_frame_8 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_28 = T.tensor_view(
                            T.ptr_of(value_16[0:0 + 64, 0:0 + 256]),
                            layout=((8 @ threads_39.d1, 2, 4 @ threads_39.d0, 2, 4 @ threads_39.d2, 32), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 256),
                        )
                        T.binary(lhs_frame_14, rhs_frame_8, dst_frame_28, kind=BinaryKind.MUL)
                    scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_6, 0.09016844005556021)
                    with scope_7 as threads_42:
                        lhs_frame_15 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_29 = T.tensor_view(
                            T.ptr_of(value_24[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_15, scalar_6, dst_frame_29, kind=BinaryKind.MUL)
                    scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_7, 0.09016844005556021)
                    with scope_7 as threads_43:
                        lhs_frame_16 = T.tensor_view(
                            T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_30 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_16, scalar_7, dst_frame_30, kind=BinaryKind.MUL)
                        lhs_frame_17 = T.tensor_view(
                            T.ptr_of(value_24[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_9 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_31 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_17, rhs_frame_9, dst_frame_31, kind=BinaryKind.SUB)
                        src_frame_12 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_32 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.unary(src_frame_12, dst_frame_32, kind=UnaryKind.EXP2)
                        src_frame_13 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_33 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_13, dst_frame_33, dtype="bf16")
                        src_frame_14 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_34 = T.tensor_view(
                            T.ptr_of(value_21[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_14, dst_frame_34)
                        for o_m_3 in range(0, 64, 64):
                            for o_n_3 in range(0, 256, 256):
                                for o_k_3 in range(0, 64, 16):
                                    acc_view_9 = T.tensor_view(
                                        T.ptr_of(value_16[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                        layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_9 = T.tensor_view(
                                        T.ptr_of(value_21[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                        layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_9 = T.tensor_view(
                                        T.ptr_of(v_ring_stages[j % 2][o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_9,
                                        lhs_view_9,
                                        rhs_view_9,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_43),
                                    )
                        lhs_frame_18 = T.tensor_view(
                            T.ptr_of(l0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_10 = T.tensor_view(
                            T.ptr_of(alpha0[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_35 = T.tensor_view(
                            T.ptr_of(value_26[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_18, rhs_frame_10, dst_frame_35, kind=BinaryKind.MUL)
                        src_frame_15 = T.tensor_view(
                            T.ptr_of(P0[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 2, 4 @ threads_43.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_36 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.reduce(
                            src_frame_15,
                            dst_frame_36,
                            axes=(-1,),
                            keepdim=True,
                            kind=ReduceKind.SUM,
                        )
                        lhs_frame_19 = T.tensor_view(
                            T.ptr_of(value_26[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        rhs_frame_11 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_37 = T.tensor_view(
                            T.ptr_of(l0_1[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_43.d1, 2, 4 @ threads_43.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            shape=(64, 1),
                        )
                        T.binary(lhs_frame_19, rhs_frame_11, dst_frame_37, kind=BinaryKind.ADD)
                    l0_1_view = T.tensor_view(
                        T.ptr_of(l0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ scope_7.d1, 2, 4 @ scope_7.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(l0_1_view, l0)
                    value_16_view = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ scope_7.d1, 2, 4 @ scope_7.d0, 2, 4 @ scope_7.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(value_16_view, O0)
                    m_new0_view = T.tensor_view(
                        T.ptr_of(m_new0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ scope_7.d1, 2, 4 @ scope_7.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(m_new0_view, m0)
                with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads_55:
                    l1_1_view = T.tensor_view(
                        T.ptr_of(l1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_55.d1, 2, 4 @ threads_55.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(l1_1_view, l1)
                    value_4_view = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_55.d1, 2, 4 @ threads_55.d0, 2, 4 @ threads_55.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(value_4_view, O1)
                    m_new1_view = T.tensor_view(
                        T.ptr_of(m_new1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_55.d1, 2, 4 @ threads_55.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(m_new1_view, m1)
            with scope[:1, :32] as scope_8:
                tile_10 = T.tensor_view(
                    T.ptr_of(k[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (1024, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_11 = T.tensor_view(
                    T.ptr_of(tile_10[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_8, layout=(32,), names=("d0",)) as threads_58:
                    T.copy_async_tensor(tile_11, k_ring_1_stages[0])
            with scope[:1, 32:64] as scope_9:
                tile_12 = T.tensor_view(
                    T.ptr_of(v[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 256]),
                    layout=((64, 1, 256), (1024, 256, 1)),
                    shape=(64, 1, 256),
                )
                tile_13 = T.tensor_view(
                    T.ptr_of(tile_12[0:0 + 64, 0:0 + 1, 0:0 + 256]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_9, layout=(32,), names=("d0",)) as threads_59:
                    T.copy_async_tensor(tile_13, v_ring_1_stages[0])
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_10:
                with scope_10 as threads_60:
                    T.fill(value_32, 0.0)
                tile_14 = T.tensor_view(
                    T.ptr_of(k_ring_1_stages[0][0:0 + 64, 0:0 + 256]),
                    layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                    shape=(256, 64),
                )
                with scope_10 as threads_61:
                    for o_m_4 in range(0, 64, 64):
                        for o_n_4 in range(0, 64, 64):
                            for o_k_4 in range(0, 256, 64):
                                acc_view_10 = T.tensor_view(
                                    T.ptr_of(value_32[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                    layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_10 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_10 = T.tensor_view(
                                    T.ptr_of(tile_14[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_10,
                                    lhs_view_10,
                                    rhs_view_10,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_61),
                                )
                                acc_view_11 = T.tensor_view(
                                    T.ptr_of(value_32[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                    layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_11 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_4:o_m_4 + 64, o_k_4 + 16:o_k_4 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_11 = T.tensor_view(
                                    T.ptr_of(tile_14[o_k_4 + 16:o_k_4 + 16 + 16, o_n_4:o_n_4 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_11,
                                    lhs_view_11,
                                    rhs_view_11,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_61),
                                )
                                acc_view_12 = T.tensor_view(
                                    T.ptr_of(value_32[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                    layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_12 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_4:o_m_4 + 64, o_k_4 + 32:o_k_4 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_12 = T.tensor_view(
                                    T.ptr_of(tile_14[o_k_4 + 32:o_k_4 + 32 + 16, o_n_4:o_n_4 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_12,
                                    lhs_view_12,
                                    rhs_view_12,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_61),
                                )
                                acc_view_13 = T.tensor_view(
                                    T.ptr_of(value_32[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                    layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_13 = T.tensor_view(
                                    T.ptr_of(q_dst0[o_m_4:o_m_4 + 64, o_k_4 + 48:o_k_4 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_13 = T.tensor_view(
                                    T.ptr_of(tile_14[o_k_4 + 48:o_k_4 + 48 + 16, o_n_4:o_n_4 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_13,
                                    lhs_view_13,
                                    rhs_view_13,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_61),
                                )
                    lhs_frame_20 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_12 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_38 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_20, rhs_frame_12, dst_frame_38, kind=BinaryKind.ADD)
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 2, 4 @ threads_61.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_39 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_16,
                        dst_frame_39,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                    lhs_frame_21 = T.tensor_view(
                        T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_13 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_40 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_21, rhs_frame_13, dst_frame_40, kind=BinaryKind.MAX)
                    lhs_frame_22 = T.tensor_view(
                        T.ptr_of(m0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_14 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_41 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_61.d1, 2, 4 @ threads_61.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_22, rhs_frame_14, dst_frame_41, kind=BinaryKind.SUB)
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 0.09016844005556021)
                with scope_10 as threads_66:
                    lhs_frame_23 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_42 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_23, scalar_8, dst_frame_42, kind=BinaryKind.MUL)
                    src_frame_17 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_43 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.unary(src_frame_17, dst_frame_43, kind=UnaryKind.EXP2)
                    lhs_frame_24 = T.tensor_view(
                        T.ptr_of(O0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 2, 4 @ threads_66.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_15 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_44 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_66.d1, 2, 4 @ threads_66.d0, 2, 4 @ threads_66.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_24, rhs_frame_15, dst_frame_44, kind=BinaryKind.MUL)
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 0.09016844005556021)
                with scope_10 as threads_69:
                    lhs_frame_25 = T.tensor_view(
                        T.ptr_of(S0[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_69.d1, 2, 4 @ threads_69.d0, 2, 4 @ threads_69.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_45 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_69.d1, 2, 4 @ threads_69.d0, 2, 4 @ threads_69.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_25, scalar_9, dst_frame_45, kind=BinaryKind.MUL)
                scalar_10 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_10, 0.09016844005556021)
                with scope_10 as threads_70:
                    lhs_frame_26 = T.tensor_view(
                        T.ptr_of(m_new0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_46 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_26, scalar_10, dst_frame_46, kind=BinaryKind.MUL)
                    lhs_frame_27 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_16 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_47 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_27, rhs_frame_16, dst_frame_47, kind=BinaryKind.SUB)
                    src_frame_18 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_48 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.unary(src_frame_18, dst_frame_48, kind=UnaryKind.EXP2)
                    src_frame_19 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_49 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.cast(src_frame_19, dst_frame_49, dtype="bf16")
                    src_frame_20 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_50 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 64, 0:0 + 64]),
                        layout=((4, 8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(src_frame_20, dst_frame_50)
                    for o_m_5 in range(0, 64, 64):
                        for o_n_5 in range(0, 256, 256):
                            for o_k_5 in range(0, 64, 16):
                                acc_view_14 = T.tensor_view(
                                    T.ptr_of(value_28[o_m_5:o_m_5 + 64, o_n_5:o_n_5 + 256]),
                                    layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 32), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 256),
                                )
                                lhs_view_14 = T.tensor_view(
                                    T.ptr_of(value_33[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                    layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 2), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 16),
                                )
                                rhs_view_14 = T.tensor_view(
                                    T.ptr_of(v_ring_1_stages[0][o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_14,
                                    lhs_view_14,
                                    rhs_view_14,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_70),
                                )
                    lhs_frame_28 = T.tensor_view(
                        T.ptr_of(l0[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_17 = T.tensor_view(
                        T.ptr_of(alpha0_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_51 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_28, rhs_frame_17, dst_frame_51, kind=BinaryKind.MUL)
                    src_frame_21 = T.tensor_view(
                        T.ptr_of(P0_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 2, 4 @ threads_70.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_52 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_21,
                        dst_frame_52,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    lhs_frame_29 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_18 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_53 = T.tensor_view(
                        T.ptr_of(l0_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_70.d1, 2, 4 @ threads_70.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_29, rhs_frame_18, dst_frame_53, kind=BinaryKind.ADD)
            with scope[:1, :32] as scope_11:
                tile_15 = T.tensor_view(
                    T.ptr_of(qkv[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, (((cta.d1 * 2) + 0) * 512) + 256:(((cta.d1 * 2) + 0) * 512) + 256 + 256]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_11, layout=(32,), names=("d0",)) as threads_79:
                    T.copy_async_tensor(tile_15, g_dst0)
                tile_16 = T.tensor_view(
                    T.ptr_of(qkv[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, (((cta.d1 * 2) + 1) * 512) + 256:(((cta.d1 * 2) + 1) * 512) + 256 + 256]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                with Mesh(scope_11, layout=(32,), names=("d0",)) as threads_80:
                    T.copy_async_tensor(tile_16, g_dst1)
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_12:
                scalar_11 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_11, 1.0)
                with scope_12 as threads_81:
                    rhs_frame_19 = T.tensor_view(
                        T.ptr_of(l0_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_54 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(scalar_11, rhs_frame_19, dst_frame_54, kind=BinaryKind.DIV)
                    lhs_frame_30 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_20 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_55 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_30, rhs_frame_20, dst_frame_55, kind=BinaryKind.MUL)
                    src_frame_22 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_56 = T.tensor_view(
                        T.ptr_of(ob0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_22, dst_frame_56, dtype="bf16")
                    src_frame_23 = T.tensor_view(
                        T.ptr_of(ob0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_57 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_81.d1, 2, 4 @ threads_81.d0, 2, 4 @ threads_81.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_23, dst_frame_57, dtype="f32")
                scalar_12 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_12, 1.0)
                with scope_12 as threads_85:
                    dst_frame_58 = T.tensor_view(
                        T.ptr_of(value_52[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_85.d1, 2, 4 @ threads_85.d0, 2, 4 @ threads_85.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(g_dst0, dst_frame_58)
                    src_frame_24 = T.tensor_view(
                        T.ptr_of(value_52[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_85.d1, 2, 4 @ threads_85.d0, 2, 4 @ threads_85.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_59 = T.tensor_view(
                        T.ptr_of(value_51[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_85.d1, 2, 4 @ threads_85.d0, 2, 4 @ threads_85.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_24, dst_frame_59, dtype="f32")
                scalar_13 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_13, -1.4426950408889634)
                with scope_12 as threads_87:
                    lhs_frame_31 = T.tensor_view(
                        T.ptr_of(value_51[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_87.d1, 2, 4 @ threads_87.d0, 2, 4 @ threads_87.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_60 = T.tensor_view(
                        T.ptr_of(value_50[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_87.d1, 2, 4 @ threads_87.d0, 2, 4 @ threads_87.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_31, scalar_13, dst_frame_60, kind=BinaryKind.MUL)
                    src_frame_25 = T.tensor_view(
                        T.ptr_of(value_50[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_87.d1, 2, 4 @ threads_87.d0, 2, 4 @ threads_87.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_61 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_87.d1, 2, 4 @ threads_87.d0, 2, 4 @ threads_87.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.unary(src_frame_25, dst_frame_61, kind=UnaryKind.EXP2)
                scalar_14 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_14, 1.0)
                with scope_12 as threads_89:
                    lhs_frame_32 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_62 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_32, scalar_14, dst_frame_62, kind=BinaryKind.ADD)
                    rhs_frame_21 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_63 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(scalar_12, rhs_frame_21, dst_frame_63, kind=BinaryKind.DIV)
                    src_frame_26 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_64 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_26, dst_frame_64, dtype="bf16")
                    src_frame_27 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_65 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_27, dst_frame_65, dtype="f32")
                    lhs_frame_33 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_22 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_66 = T.tensor_view(
                        T.ptr_of(value_41[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_33, rhs_frame_22, dst_frame_66, kind=BinaryKind.MUL)
                    src_frame_28 = T.tensor_view(
                        T.ptr_of(value_41[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_67 = T.tensor_view(
                        T.ptr_of(y0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_28, dst_frame_67, dtype="bf16")
                    src_frame_29 = T.tensor_view(
                        T.ptr_of(y0[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_89.d1, 2, 4 @ threads_89.d0, 2, 4 @ threads_89.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(src_frame_29, value_40)
            with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_13:
                with scope_13 as threads_96:
                    T.fill(value_57, 0.0)
                tile_17 = T.tensor_view(
                    T.ptr_of(k_ring_1_stages[0][0:0 + 64, 0:0 + 256]),
                    layout=Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                    shape=(256, 64),
                )
                with scope_13 as threads_97:
                    for o_m_6 in range(0, 64, 64):
                        for o_n_6 in range(0, 64, 64):
                            for o_k_6 in range(0, 256, 64):
                                acc_view_15 = T.tensor_view(
                                    T.ptr_of(value_57[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                    layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_15 = T.tensor_view(
                                    T.ptr_of(q_dst1[o_m_6:o_m_6 + 64, o_k_6:o_k_6 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_15 = T.tensor_view(
                                    T.ptr_of(tile_17[o_k_6:o_k_6 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_15,
                                    lhs_view_15,
                                    rhs_view_15,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_97),
                                )
                                acc_view_16 = T.tensor_view(
                                    T.ptr_of(value_57[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                    layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_16 = T.tensor_view(
                                    T.ptr_of(q_dst1[o_m_6:o_m_6 + 64, o_k_6 + 16:o_k_6 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_16 = T.tensor_view(
                                    T.ptr_of(tile_17[o_k_6 + 16:o_k_6 + 16 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_16,
                                    lhs_view_16,
                                    rhs_view_16,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_97),
                                )
                                acc_view_17 = T.tensor_view(
                                    T.ptr_of(value_57[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                    layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_17 = T.tensor_view(
                                    T.ptr_of(q_dst1[o_m_6:o_m_6 + 64, o_k_6 + 32:o_k_6 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_17 = T.tensor_view(
                                    T.ptr_of(tile_17[o_k_6 + 32:o_k_6 + 32 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_17,
                                    lhs_view_17,
                                    rhs_view_17,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_97),
                                )
                                acc_view_18 = T.tensor_view(
                                    T.ptr_of(value_57[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                    layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_18 = T.tensor_view(
                                    T.ptr_of(q_dst1[o_m_6:o_m_6 + 64, o_k_6 + 48:o_k_6 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_18 = T.tensor_view(
                                    T.ptr_of(tile_17[o_k_6 + 48:o_k_6 + 48 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_18,
                                    lhs_view_18,
                                    rhs_view_18,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_97),
                                )
                    lhs_frame_34 = T.tensor_view(
                        T.ptr_of(value_57[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_23 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_68 = T.tensor_view(
                        T.ptr_of(S1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_34, rhs_frame_23, dst_frame_68, kind=BinaryKind.ADD)
                    src_frame_30 = T.tensor_view(
                        T.ptr_of(S1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 2, 4 @ threads_97.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_69 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_30,
                        dst_frame_69,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                    lhs_frame_35 = T.tensor_view(
                        T.ptr_of(m1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_24 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_70 = T.tensor_view(
                        T.ptr_of(m_new1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_35, rhs_frame_24, dst_frame_70, kind=BinaryKind.MAX)
                    lhs_frame_36 = T.tensor_view(
                        T.ptr_of(m1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_25 = T.tensor_view(
                        T.ptr_of(m_new1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_71 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_97.d1, 2, 4 @ threads_97.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_36, rhs_frame_25, dst_frame_71, kind=BinaryKind.SUB)
                scalar_15 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_15, 0.09016844005556021)
                with scope_13 as threads_102:
                    lhs_frame_37 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_72 = T.tensor_view(
                        T.ptr_of(value_54[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_37, scalar_15, dst_frame_72, kind=BinaryKind.MUL)
                    src_frame_31 = T.tensor_view(
                        T.ptr_of(value_54[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_73 = T.tensor_view(
                        T.ptr_of(alpha1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.unary(src_frame_31, dst_frame_73, kind=UnaryKind.EXP2)
                    lhs_frame_38 = T.tensor_view(
                        T.ptr_of(O1[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 2, 4 @ threads_102.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_26 = T.tensor_view(
                        T.ptr_of(alpha1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_74 = T.tensor_view(
                        T.ptr_of(value_53[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_102.d1, 2, 4 @ threads_102.d0, 2, 4 @ threads_102.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_38, rhs_frame_26, dst_frame_74, kind=BinaryKind.MUL)
                scalar_16 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_16, 0.09016844005556021)
                with scope_13 as threads_105:
                    lhs_frame_39 = T.tensor_view(
                        T.ptr_of(S1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_105.d1, 2, 4 @ threads_105.d0, 2, 4 @ threads_105.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_75 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_105.d1, 2, 4 @ threads_105.d0, 2, 4 @ threads_105.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_39, scalar_16, dst_frame_75, kind=BinaryKind.MUL)
                scalar_17 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_17, 0.09016844005556021)
                with scope_13 as threads_106:
                    lhs_frame_40 = T.tensor_view(
                        T.ptr_of(m_new1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_76 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_40, scalar_17, dst_frame_76, kind=BinaryKind.MUL)
                    lhs_frame_41 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_27 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_77 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_41, rhs_frame_27, dst_frame_77, kind=BinaryKind.SUB)
                    src_frame_32 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_78 = T.tensor_view(
                        T.ptr_of(P1_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.unary(src_frame_32, dst_frame_78, kind=UnaryKind.EXP2)
                    src_frame_33 = T.tensor_view(
                        T.ptr_of(P1_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_79 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.cast(src_frame_33, dst_frame_79, dtype="bf16")
                    src_frame_34 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_80 = T.tensor_view(
                        T.ptr_of(value_58[0:0 + 64, 0:0 + 64]),
                        layout=((4, 8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(src_frame_34, dst_frame_80)
                    for o_m_7 in range(0, 64, 64):
                        for o_n_7 in range(0, 256, 256):
                            for o_k_7 in range(0, 64, 16):
                                acc_view_19 = T.tensor_view(
                                    T.ptr_of(value_53[o_m_7:o_m_7 + 64, o_n_7:o_n_7 + 256]),
                                    layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 32), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 256),
                                )
                                lhs_view_19 = T.tensor_view(
                                    T.ptr_of(value_58[o_m_7:o_m_7 + 64, o_k_7:o_k_7 + 16]),
                                    layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 2), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 16),
                                )
                                rhs_view_19 = T.tensor_view(
                                    T.ptr_of(v_ring_1_stages[0][o_k_7:o_k_7 + 16, o_n_7:o_n_7 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_19,
                                    lhs_view_19,
                                    rhs_view_19,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_106),
                                )
                    lhs_frame_42 = T.tensor_view(
                        T.ptr_of(l1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_28 = T.tensor_view(
                        T.ptr_of(alpha1_1[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_81 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_42, rhs_frame_28, dst_frame_81, kind=BinaryKind.MUL)
                    src_frame_35 = T.tensor_view(
                        T.ptr_of(P1_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 2, 4 @ threads_106.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_82 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_35,
                        dst_frame_82,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    lhs_frame_43 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_29 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_83 = T.tensor_view(
                        T.ptr_of(l1_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_106.d1, 2, 4 @ threads_106.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_43, rhs_frame_29, dst_frame_83, kind=BinaryKind.ADD)
                scalar_18 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_18, 1.0)
                with scope_13 as threads_115:
                    rhs_frame_30 = T.tensor_view(
                        T.ptr_of(l1_2[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_84 = T.tensor_view(
                        T.ptr_of(value_69[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(scalar_18, rhs_frame_30, dst_frame_84, kind=BinaryKind.DIV)
                    lhs_frame_44 = T.tensor_view(
                        T.ptr_of(value_53[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_31 = T.tensor_view(
                        T.ptr_of(value_69[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_85 = T.tensor_view(
                        T.ptr_of(value_68[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_44, rhs_frame_31, dst_frame_85, kind=BinaryKind.MUL)
                    src_frame_36 = T.tensor_view(
                        T.ptr_of(value_68[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_86 = T.tensor_view(
                        T.ptr_of(ob1[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_36, dst_frame_86, dtype="bf16")
                    src_frame_37 = T.tensor_view(
                        T.ptr_of(ob1[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_87 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_115.d1, 2, 4 @ threads_115.d0, 2, 4 @ threads_115.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_37, dst_frame_87, dtype="f32")
                scalar_19 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_19, 1.0)
                with scope_13 as threads_119:
                    dst_frame_88 = T.tensor_view(
                        T.ptr_of(value_77[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_119.d1, 2, 4 @ threads_119.d0, 2, 4 @ threads_119.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(g_dst1, dst_frame_88)
                    src_frame_38 = T.tensor_view(
                        T.ptr_of(value_77[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_119.d1, 2, 4 @ threads_119.d0, 2, 4 @ threads_119.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_89 = T.tensor_view(
                        T.ptr_of(value_76[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_119.d1, 2, 4 @ threads_119.d0, 2, 4 @ threads_119.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_38, dst_frame_89, dtype="f32")
                scalar_20 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_20, -1.4426950408889634)
                with scope_13 as threads_121:
                    lhs_frame_45 = T.tensor_view(
                        T.ptr_of(value_76[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_121.d1, 2, 4 @ threads_121.d0, 2, 4 @ threads_121.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_90 = T.tensor_view(
                        T.ptr_of(value_75[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_121.d1, 2, 4 @ threads_121.d0, 2, 4 @ threads_121.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_45, scalar_20, dst_frame_90, kind=BinaryKind.MUL)
                    src_frame_39 = T.tensor_view(
                        T.ptr_of(value_75[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_121.d1, 2, 4 @ threads_121.d0, 2, 4 @ threads_121.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_91 = T.tensor_view(
                        T.ptr_of(value_74[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_121.d1, 2, 4 @ threads_121.d0, 2, 4 @ threads_121.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.unary(src_frame_39, dst_frame_91, kind=UnaryKind.EXP2)
                scalar_21 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_21, 1.0)
                with scope_13 as threads_123:
                    lhs_frame_46 = T.tensor_view(
                        T.ptr_of(value_74[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_92 = T.tensor_view(
                        T.ptr_of(value_73[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_46, scalar_21, dst_frame_92, kind=BinaryKind.ADD)
                    rhs_frame_32 = T.tensor_view(
                        T.ptr_of(value_73[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_93 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(scalar_19, rhs_frame_32, dst_frame_93, kind=BinaryKind.DIV)
                    src_frame_40 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_94 = T.tensor_view(
                        T.ptr_of(value_71[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_40, dst_frame_94, dtype="bf16")
                    src_frame_41 = T.tensor_view(
                        T.ptr_of(value_71[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_95 = T.tensor_view(
                        T.ptr_of(value_70[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_41, dst_frame_95, dtype="f32")
                    lhs_frame_47 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    rhs_frame_33 = T.tensor_view(
                        T.ptr_of(value_70[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_96 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.binary(lhs_frame_47, rhs_frame_33, dst_frame_96, kind=BinaryKind.MUL)
                    src_frame_42 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    dst_frame_97 = T.tensor_view(
                        T.ptr_of(y1[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.cast(src_frame_42, dst_frame_97, dtype="bf16")
                    src_frame_43 = T.tensor_view(
                        T.ptr_of(y1[0:0 + 64, 0:0 + 256]),
                        layout=((8 @ threads_123.d1, 2, 4 @ threads_123.d0, 2, 4 @ threads_123.d2, 32), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 256),
                    )
                    T.copy(src_frame_43, value_65)
            with scope[:1, :32] as scope_15:
                with Mesh(scope_15, layout=(32,), names=("d0",)) as threads_130:
                    T.copy_async_tensor(value_40, value_78_1)
                window = T.tensor_view(
                    T.ptr_of(out[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, ((cta.d1 * 2) + 0) * 256:((cta.d1 * 2) + 0) * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_78_1, window)
                with Mesh(scope_15, layout=(32,), names=("d0",)) as threads_131:
                    T.copy_async_tensor(value_65, value_79_1)
                window_1 = T.tensor_view(
                    T.ptr_of(out[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, ((cta.d1 * 2) + 1) * 256:((cta.d1 * 2) + 1) * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_79_1, window_1)
