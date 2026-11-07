"""Four CTAs store row windows through tensor maps, then gather rows that data chooses.

Every CTA stores its own window of a global table and then gathers rows of the
whole table by index. Which rows it reads is data, so the plan cannot prove it
avoids a window another CTA is still storing.
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
class TmaStoreThenGather:
    @func
    def store(
        x: Tensor[(CTAS * ROWS, COLS), "f32"],
        idx: Tensor[(ROWS,), "i64"],
    ) -> Tensor[(CTAS * ROWS, COLS), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                row = cta.block * ROWS
                staged_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                staged = tf.schedule(
                    (x[row:row + ROWS, :], staged_dst), op=T.copy_async_tensor()
                )
                tile_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                table = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
                table = tf.insert_slice(table, tile, (row, 0))
                picked_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                picked = tf.schedule((table, picked_dst, idx), op=T.copy_async(fill=0))
                out_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                out_tile = tf.schedule((picked, out_dst), op=T.copy_async_tensor())
                out = tf.empty(Tensor[(CTAS * ROWS, COLS), "f32"])
                return tf.insert_slice(out, out_tile, (row, 0))
