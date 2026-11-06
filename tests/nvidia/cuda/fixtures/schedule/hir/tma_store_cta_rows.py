"""A two by two CTA grid stores row windows chosen by the first CTA axis alone.

The two CTAs of one grid row store the same window; the relation never names
the second axis that tells them apart.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

ROWS = 64
COLS = 64


@module(
    entry="store",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 4), Topology("thread", 32)),
)
class TmaStoreCtaRows:
    @func
    def store(x: Tensor[(2 * ROWS, COLS), "f32"]) -> Tensor[(2 * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(2, 2), names=("row", "column")) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                out = tf.empty(Tensor[(2 * ROWS, COLS), "f32"])
                for row in range(cta.row * ROWS, 2 * ROWS, 2 * ROWS):
                    staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                    staged = tf.schedule(
                        (x[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                    )
                    tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out
