from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.rtx_6000_ada"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class Sm89E4m3Gemm:
    @prim_func(target=CudaTarget("nvidia.rtx_6000_ada"))
    def gemm(
        a: Tensor[(16, 128), "fp8e4m3"], b: Tensor[(128, 8), "fp8e4m3", ((128, 8), (1, 128))], out: Tensor[(16, 8), "f32"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh(
                (Topology("thread", 32),), layout=((4, 8), (1, 4)), names=("warp", "lane")
            ) as scope:
                T.fill(out, 0.0)
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 8),
                        "f32",
                        ((2, 8 @ scope.lane, 4 @ scope.warp, 2), (64, 8, 2, 1)),
                        "rmem",
                    ]
                )
                T.fill(acc, 0.0)
                for k in range(0, 128, 32):
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]), layout=((16, 32), (128, 1)), shape=(16, 32)
                    )
                    lhs_stage = T.tensor_view(
                        0, dtype="fp8e4m3", storage="smem", layout=(16, 32), shape=(16, 32)
                    )
                    T.copy_async(value, lhs_stage)
                    lhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 32),
                            "fp8e4m3",
                            ((2, 8 @ scope.lane, 2, 4 @ scope.warp, 4), (256, 32, 16, 4, 1)),
                            "rmem",
                        ]
                    )
                    T.copy(lhs_stage, lhs_frag_dst)
                    value_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=((32, 8), (1, 128)), shape=(32, 8)
                    )
                    rhs_stage = T.tensor_view(
                        0, dtype="fp8e4m3", storage="smem", layout=((32, 8), (1, 32)), shape=(32, 8)
                    )
                    T.copy_async(value_1, rhs_stage)
                    rhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8),
                            "fp8e4m3",
                            ((2, 4 @ scope.warp, 4, 8 @ scope.lane), (128, 32, 8, 1)),
                            "rmem",
                        ]
                    )
                    T.copy(rhs_stage, rhs_frag_dst)
                    acc_view = T.tensor_view(
                        T.ptr_of(acc[0:0 + 16, 0:0 + 8]),
                        layout=((2, 8 @ scope.lane, 4 @ scope.warp, 2), (64, 8, 2, 1)),
                        shape=(16, 8),
                    )
                    lhs_view = lhs_frag_dst[0:0 + 16, 0:0 + 32]
                    rhs_view = rhs_frag_dst[0:0 + 32, 0:0 + 8]
                    T.tiled_mma(acc_view, lhs_view, rhs_view, atom=T.cuda.sm89.Mma())
                tile = T.tensor_view(
                    T.ptr_of(acc[0:0 + 16, 0:0 + 8]),
                    layout=((2, 8 @ scope.lane, 4 @ scope.warp, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                T.copy(tile, out)
