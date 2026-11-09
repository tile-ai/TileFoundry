"""Four CTAs each store a row window through a tensor map and copy out their neighbour's.

Every CTA stores its own window of a global exchange buffer, then copies the
next CTA's window into its output rows; nothing orders that read after the
other CTA's store.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 64
COLS = 64
CTAS = 4


@module(
    entry="store",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 32)),
)
class TmaStoreNeighborExchange:
    @func
    def store(x: Tensor[(CTAS * ROWS, COLS), "f32"]) -> Tensor[(CTAS * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                row = cta.block * ROWS
                staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                staged = tf.schedule(
                    (x[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                )
                tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                exchange = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
                exchange = tf.insert_slice(exchange, tile, (row, 0))
                neighbor = ((cta.block + 1) % CTAS) * ROWS
                out = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
                return tf.insert_slice(out, exchange[neighbor:neighbor + ROWS, :], (row, 0))
