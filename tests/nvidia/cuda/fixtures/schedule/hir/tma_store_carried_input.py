"""Four CTAs each store their own row window into a loop-carried copy of the input.

The loop starts its carried output from the input, so every CTA copies all of
it on entry while the other CTAs' tensor-map stores land in their windows.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

ROWS = 64
COLS = 64
CTAS = 4


@module(
    entry="store",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 32)),
)
class TmaStoreCarriedInput:
    @func
    def store(
        x: Tensor[(CTAS * ROWS, COLS), "f32"],
        y: Tensor[(CTAS * ROWS, COLS), "f32"],
    ) -> Tensor[(CTAS * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                out = x
                for row in range(cta.block * ROWS, CTAS * ROWS, CTAS * ROWS):
                    staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                    staged = tf.schedule(
                        (y[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                    )
                    tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out
