# analysis target=nvidia.h200_sxm module=SM80_MMA_LDMATRIX function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r1.50KB/w256B@logical,r1.50KB/w256B@total,r1.50KB/w256B@cta,r1.50KB/w8B@thread;rmem:r3.33KB/w3.25KB@logical,r3.33KB/w3.25KB@total,r3.33KB/w3.25KB@cta,r168B/w88B@thread;smem:r1.50KB/w1.50KB@logical,r1.50KB/w1.50KB@total,r1.50KB/w1.50KB@cta,r48B/w1.50KB@thread footprint=a:512B;b:256B;v16:100:256B footprint-precision=exact peak=gmem:1.75KB;rmem:1.25KB;smem:1.50KB persistent=gmem:1.50KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(a: Tensor[(16, 32), "bf16"], b: Tensor[(32, 8), "bf16"], out: Tensor[(16, 8), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 64),), layout=(2, 32), names=("d0", "d1")) as scope:
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
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_1:
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
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 16, k:k + 16]),
                        layout=((16, 16), (32, 1)),
                        shape=(16, 16),
                    )
                    T.copy_async_tensor(tile, lhs_ring_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 8]), layout=(16, 8), shape=(16, 8)
                    )
                    T.copy_async_tensor(tile_1, rhs_ring_stages[(k // 16) % 2])
                with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_3:
                    lhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16),
                            "bf16",
                            ((2, 8 @ scope_3.d1, 2, 4 @ scope_3.d0, 2), (128, 16, 8, 2, 1)),
                            "rmem",
                        ]
                    )
                    T.ldmatrix(lhs_ring_stages[(k // 16) % 2], lhs_frag_dst)
                    rhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 8),
                            "bf16",
                            ((2, 4 @ scope_3.d0, 2, 8 @ scope_3.d1), (64, 16, 8, 1)),
                            "rmem",
                        ]
                    )
                    T.copy(rhs_ring_stages[(k // 16) % 2], rhs_frag_dst)
                    for o_m in range(0, 16, 16):
                        for o_n in range(0, 8, 8):
                            for o_k in range(0, 16, 16):
                                T.tiled_mma(
                                    acc,
                                    lhs_frag_dst,
                                    rhs_frag_dst,
                                    atom=T.cuda.sm80.Mma(),
                                )
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 8),
                        "bf16",
                        ((2, 8 @ scope_4.d1, 4 @ scope_4.d0, 2), (64, 8, 2, 1)),
                        "rmem",
                    ]
                )
                T.cast(acc, result, dtype="bf16")
                window = T.tensor_view(
                    T.ptr_of(out[0:0 + 16, 0:0 + 8]), layout=(16, 8), shape=(16, 8)
                )
                T.copy(result, window)
