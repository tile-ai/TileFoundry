"""One loader warp and one compute warpgroup share a loop over K.

Each slice's copy overwrites the tiles the last slice's WGMMA reads, so the wait
for the WGMMA group belongs in the loop body, over all 160 threads. Five warps are
not whole warpgroups, so the WGMMA completes right after it issues instead.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 160)),
)
class WgmmaPartial:
    @func
    def gemm(
        a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 16), "bf16"]
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(5, 32), names=("warp", "lane")) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K
                )
                with Mesh(threads[0:4, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as c:
                    acc = tf.zeros(Tensor[
                        (64, 16), "f32",
                        ((4 @ c.warp, 2, 8 @ c.lane8, 2, 4 @ c.lane4, 2,), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ])
                for k in tf.tile(32, 16):
                    with threads[4, :] as _loader:
                        lhs = tf.schedule(
                            (a[:, k], tf.empty(Tensor[
                                (64, 16), "bf16",
                                Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), "smem",
                            ])),
                            op=T.copy_async(),
                        )
                        rhs = tf.schedule(
                            (b[k, :], tf.empty(Tensor[
                                (16, 16), "bf16",
                                Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), "smem",
                            ])),
                            op=T.copy_async(),
                        )
                    with Mesh(threads[0:4, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c:
                        acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=wgmma))
                output = tf.zeros(Tensor[(64, 16), "f32"])
                with Mesh(threads[0:4, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c:
                    result = tf.insert_slice(output, acc, (0, 0))
                return result
