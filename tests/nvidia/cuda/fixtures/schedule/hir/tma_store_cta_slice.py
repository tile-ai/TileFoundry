"""Only the first two of four CTAs run a sliced mesh; each stores its own row window.

The slice keeps CTAs 2 and 3 out, so the two windows come from two CTAs alone.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 64
COLS = 64


@module(
    entry="store",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 4), Topology("thread", 32)),
)
class TmaStoreCtaSlice:
    @func
    def store(x: Tensor[(2 * ROWS, COLS), "f32"]) -> Tensor[(2 * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(4,), names=("block",)) as ctas:
            with ctas[:2] as first:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                    out = tf.empty(Tensor[(2 * ROWS, COLS), "f32"])
                    for row in range(first.block * ROWS, 2 * ROWS, 2 * ROWS):
                        staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                        staged = tf.schedule(
                            (x[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                        )
                        tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                        tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (row, 0))
                    return out
