"""Four CTAs call a retained callee that round-trips a tile through its own global scratch.

The callee stores the tile into the scratch through a tensor map and reads it
back into registers, so no argument or result of the call names the scratch.
Every CTA writes the one scratch.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 32
COLS = 8
CTAS = 4


@module(
    entry="kernel",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 32)),
)
class TmaScratchInCallee:
    @func
    def round_trip(
        x: Tensor[(ROWS, COLS), "bf16"],
    ) -> Tensor[(ROWS, COLS), "bf16", (ROWS, COLS), "rmem"]:
        with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
            staged = tf.schedule(
                (x, tf.empty(Tensor[(ROWS, COLS), "bf16", (ROWS, COLS), "smem"])),
                op=T.copy_async_tensor(),
            )
            scratch = tf.schedule(
                (staged, tf.empty(Tensor[(ROWS, COLS), "bf16"])), op=T.copy_async_tensor()
            )
            return tf.schedule(
                (scratch, tf.empty(Tensor[(ROWS, COLS), "bf16", (ROWS, COLS), "rmem"])),
                op=T.copy(),
            )

    @func
    def kernel(x: Tensor[(ROWS, COLS), "bf16"]) -> Tensor[(CTAS * ROWS, COLS), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            out = tf.empty(Tensor[(CTAS * ROWS, COLS), "bf16"])
            tile = round_trip(x)
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                stored = tf.schedule(
                    (tile, tf.empty(Tensor[(ROWS, COLS), "bf16", (ROWS, COLS)])), op=T.copy()
                )
                out = tf.insert_slice(out, stored, (cta.block * ROWS, 0))
            return out
