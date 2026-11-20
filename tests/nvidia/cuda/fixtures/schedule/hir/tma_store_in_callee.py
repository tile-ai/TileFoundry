"""Four CTAs call a retained callee that stores a row window through a tensor map.

The callee runs in a thread scope only. Its window and its shared-memory tile differ
per CTA, so they name the class mesh's ``block`` axis Varying: a topology outside the
callee's scope that holds neither Split nor Partial.
"""

from __future__ import annotations

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
class TmaStoreInCallee:
    cta = Mesh(("cta",), (CTAS,), names=("block",))

    @func
    def window(x: Tensor[(ROWS, COLS), "f32", (ROWS, COLS) @ {cta.block @ V()}]) -> Tensor[(ROWS, COLS), "f32"]:
        with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
            staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS) @ {cta.block @ V()}, "smem"])
            staged = tf.schedule((x, staged_dst), op=T.copy_async_tensor())
            tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
            return tf.schedule((staged, tile_dst), op=T.copy_async_tensor())

    @func
    def store(x: Tensor[(CTAS * ROWS, COLS), "f32"]) -> Tensor[(CTAS * ROWS, COLS), "f32"]:
        with cta:
            out = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
            for row in range(cta.block * ROWS, CTAS * ROWS, CTAS * ROWS):
                tile = window(x[row:row + ROWS, :])
                out = tf.insert_slice(out, tile, (row, 0))
            return out
