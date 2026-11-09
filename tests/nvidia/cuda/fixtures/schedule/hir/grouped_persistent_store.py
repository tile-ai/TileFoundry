"""Persistent CTAs store output tiles in the grouped order of the schedule tutorial.

The window offsets divide by a product of constants. Each tile still lands in
its own window of the output, with no global scratch between them.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 64
COLS = 64
CTAS = 4
GM = 4
GN = 3
GROUP_M = 2


@module(
    entry="store",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 64)),
)
class GroupedPersistentStore:
    @func
    def store(x: Tensor[(ROWS, COLS), "f32"]) -> Tensor[(GM * ROWS, GN * COLS), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                out = tf.empty(Tensor[(GM * ROWS, GN * COLS), "f32"])
                for task in range(cta.block, GM * GN, CTAS):
                    g = task // (GN * GROUP_M)
                    bn = (task // GROUP_M) % GN
                    mi = task % GROUP_M
                    m = (g * GROUP_M + mi) * ROWS
                    n = bn * COLS
                    with threads[0, :32] as _loader:
                        staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                        staged = tf.schedule((x, staged_dst), op=T.copy_async_tensor())
                    with threads[1, :32] as _storer:
                        tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                        tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (m, n))
                return out
