"""Each of 8 threads lands its own row of x through cp.async, then reads one column in halves.

Thread r copies row r only, so no two threads write the same bytes; reading column r
crosses threads, so it needs the wait and the barrier the sync plan adds.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.rtx_6000_ada"),
    topologies=(Topology("cta", 1), Topology("thread", 8)),
)
class RowWindows:
    @func
    def gemm(x: Tensor[(8, 8), "f32"]) -> Tensor[(8, 8), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(8,), names=("row",)) as threads:
                r = threads.row
                stage = tf.empty(Tensor[(8, 8), "f32", Layout((8, 8), (8, 1)), "smem"])
                landed = tf.schedule((x[r : r + 1, :], stage[r : r + 1, :]), op=T.copy_async())
                staged = tf.insert_slice(stage, landed, (r, 0))
                out = tf.zeros(Tensor[(8, 8), "f32"])
                for h in range(2):
                    half = tf.schedule(
                        (
                            staged[h * 4 : h * 4 + 4, r : r + 1],
                            tf.empty(Tensor[(4, 1), "f32", (4, 1), "rmem"]),
                        ),
                        op=T.copy(),
                    )
                    out = tf.insert_slice(out, half, (h * 4, r))
                return out
