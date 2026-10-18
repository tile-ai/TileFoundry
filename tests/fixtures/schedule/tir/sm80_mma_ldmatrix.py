# analysis target=nvidia.h200_sxm module=SM80_MMA_LDMATRIX function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r1.50KB/w256B@logical,r1.50KB/w256B@total,r1.50KB/w256B@cta,r1.50KB/w8B@thread;rmem:r3.33KB/w3.25KB@logical,r3.33KB/w3.25KB@total,r3.33KB/w3.25KB@cta,r168B/w88B@thread;smem:r1.50KB/w1.50KB@logical,r1.50KB/w1.50KB@total,r1.50KB/w1.50KB@cta,r48B/w1.50KB@thread footprint=a:512B;b:256B;v16:100:256B footprint-precision=exact peak=gmem:1.75KB;rmem:1.25KB;smem:1.50KB persistent=gmem:1.50KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(a: Tensor[(16, 32), "bf16"], b: Tensor[(32, 8), "bf16"], out: Tensor[(16, 8), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 64),), layout=(2, 32), names=("role", "participant")
        ) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 8, 4, 2), (64, 8, 2, 1)),
                        attrs=(S(2), S(1)),
                        mesh=Mesh((Topology("thread", 64),), layout=ComposedLayout(
                            inner=None,
                            offset=32,
                            outer=Layout((4, 8), (1, 4)),
                        ), names=("warp", "lane")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("warp", "lane")) as scope_1:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(512, dtype="bf16", storage="smem", layout=(16, 16), shape=(16, 16)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=(16, 16), shape=(16, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=(16, 8), shape=(16, 8)),
                T.tensor_view(256, dtype="bf16", storage="smem", layout=(16, 8), shape=(16, 8)),
            )
            for k in range(0, 32, 16):
                with scope[:1] as scope_2:
                    value = T.tensor_view(T.ptr_of(a), layout=(16, 32), shape=(16, 32))
                    value_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 1, k:k + 1]),
                        layout=((16, 16), (32, 1)),
                        shape=(16, 16),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=(16, 16),
                        shape=(16, 16),
                    )
                    T.copy_async_tensor(value_1, value_2)
                    value_3 = T.tensor_view(T.ptr_of(b), layout=(32, 8), shape=(32, 8))
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[k:k + 1, 0:0 + 1]), layout=(16, 8), shape=(16, 8)
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]), layout=(16, 8), shape=(16, 8)
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=((4, 8), (1, 4)), names=("warp", "lane")
                ) as scope_3:
                    lhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16),
                            "bf16",
                            ((2, 8 @ scope_3.lane, 2, 4 @ scope_3.warp, 2), (128, 16, 8, 2, 1)),
                            "rmem",
                        ]
                    )
                    value_6 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=(16, 16),
                        shape=(16, 16),
                    )
                    value_7 = T.tensor_view(T.ptr_of(value_6), layout=(16, 16), shape=(16, 16))
                    value_8 = T.tensor_view(T.ptr_of(value_7), layout=(16, 16), shape=(16, 16))
                    value_9 = T.tensor_view(T.ptr_of(value_8), layout=(16, 16), shape=(16, 16))
                    value_10 = T.tensor_view(T.ptr_of(value_9), layout=(16, 16), shape=(16, 16))
                    value_11 = T.tensor_view(
                        T.ptr_of(lhs_frag_dst),
                        layout=((2, 8 @ scope_3.lane, 2, 4 @ scope_3.warp, 2), (128, 16, 8, 2, 1)),
                        shape=(16, 16),
                    )
                    T.ldmatrix(value_10, value_11)
                    rhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 8),
                            "bf16",
                            ((2, 4 @ scope_3.warp, 2, 8 @ scope_3.lane), (64, 16, 8, 1)),
                            "rmem",
                        ]
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]), layout=(16, 8), shape=(16, 8)
                    )
                    value_13 = T.tensor_view(T.ptr_of(value_12), layout=(16, 8), shape=(16, 8))
                    value_14 = T.tensor_view(T.ptr_of(value_13), layout=(16, 8), shape=(16, 8))
                    value_15 = T.tensor_view(T.ptr_of(value_14), layout=(16, 8), shape=(16, 8))
                    value_16 = T.tensor_view(T.ptr_of(value_15), layout=(16, 8), shape=(16, 8))
                    value_17 = T.tensor_view(
                        T.ptr_of(rhs_frag_dst),
                        layout=((2, 4 @ scope_3.warp, 2, 8 @ scope_3.lane), (64, 16, 8, 1)),
                        shape=(16, 8),
                    )
                    T.copy(value_16, value_17)
                    value_18 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2, 8 @ scope_3.lane, 4 @ scope_3.warp, 2), (64, 8, 2, 1)),
                        shape=(16, 8),
                    )
                    value_19 = T.tensor_view(
                        T.ptr_of(value_18),
                        layout=((2, 8 @ scope_3.lane, 4 @ scope_3.warp, 2), (64, 8, 2, 1)),
                        shape=(16, 8),
                    )
                    value_20 = T.tensor_view(
                        T.ptr_of(value_19),
                        layout=((2, 8 @ scope_3.lane, 4 @ scope_3.warp, 2), (64, 8, 2, 1)),
                        shape=(16, 8),
                    )
                    acc_view = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 16, 0:0 + 8]),
                        layout=((2, 8 @ scope_3.lane, 4 @ scope_3.warp, 2), (64, 8, 2, 1)),
                        shape=(16, 8),
                    )
                    T.tiled_mma(acc_view, value_11, value_17, atom=T.cuda.sm80.Mma())
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("warp", "lane")) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 8),
                        "bf16",
                        ((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                        "rmem",
                    ]
                )
                value_21 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                value_24 = T.tensor_view(
                    T.ptr_of(value_23),
                    layout=((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_24[0:0 + 16, 0:0 + 8]),
                    layout=((2, 8 @ scope_4.lane, 4 @ scope_4.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                T.cast(src_view, result, dtype="bf16")
                value_25 = T.tensor_view(T.ptr_of(out), layout=(16, 8), shape=(16, 8))
                window = T.tensor_view(
                    T.ptr_of(value_25[0:0 + 16, 0:0 + 8]), layout=(16, 8), shape=(16, 8)
                )
                T.copy(result, window)
