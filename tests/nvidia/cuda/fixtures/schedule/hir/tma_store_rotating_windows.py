"""Four CTAs store rotating row windows: apart in each iteration, shared across iterations."""

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
class TmaStoreRotatingWindows:
    @func
    def store(x: Tensor[(ROWS, COLS), "f32"]) -> Tensor[(CTAS * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                out = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
                staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                staged = tf.schedule((x, staged_dst), op=T.copy_async_tensor())
                for turn in range(2):
                    row = (cta.block + turn) % CTAS * ROWS
                    tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out
