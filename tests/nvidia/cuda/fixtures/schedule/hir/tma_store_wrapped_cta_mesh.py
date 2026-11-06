"""Four CTAs run a two-CTA mesh, so CTAs 0 and 2, and 1 and 3, store the same row window.

A mesh smaller than its topology repeats: each CTA reads its coordinate modulo
the mesh, and no CTA is left out.
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
class TmaStoreWrappedCtaMesh:
    @func
    def store(x: Tensor[(2 * ROWS, COLS), "f32"]) -> Tensor[(2 * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("half",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                out = tf.empty(Tensor[(2 * ROWS, COLS), "f32"])
                for row in range(cta.half * ROWS, 2 * ROWS, 2 * ROWS):
                    staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                    staged = tf.schedule(
                        (x[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                    )
                    tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out
