"""Four CTAs build two updated copies of one input, then load each copy's untouched tail.

At most one copy can update the input in place, so every CTA copies all of the
input into the other one before writing its own rows. The tail rows each CTA
then loads through a tensor map are rows the other CTAs are still copying.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 64
COLS = 64
CTAS = 4
TAIL = CTAS * ROWS


@module(
    entry="update",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 32)),
)
class TmaLoadFromInputCopies:
    @func
    def update(
        x: Tensor[(TAIL + ROWS, COLS), "f32"],
        y: Tensor[(TAIL, COLS), "f32"],
    ) -> tuple[
        Tensor[(TAIL + ROWS, COLS), "f32"],
        Tensor[(TAIL + ROWS, COLS), "f32"],
        Tensor[(2 * TAIL, COLS), "f32"],
    ]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                row = cta.block * ROWS
                first = tf.insert_slice(x, y[row:row + ROWS, :], (row, 0))
                second = tf.insert_slice(x, y[row:row + ROWS, :], (row, 0))
                tails = tf.empty(Tensor[(2 * TAIL, COLS), "f32"])

                first_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                first_staged = tf.schedule(
                    (first[TAIL:TAIL + ROWS, :], first_dst), op=T.copy_async_tensor()
                )
                first_tail_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                first_tail = tf.schedule((first_staged, first_tail_dst), op=T.copy_async_tensor())
                tails = tf.insert_slice(tails, first_tail, (row, 0))

                second_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS), "smem"])
                second_staged = tf.schedule(
                    (second[TAIL:TAIL + ROWS, :], second_dst), op=T.copy_async_tensor()
                )
                second_tail_dst = tf.empty(Tensor[(ROWS, COLS), "f32", (ROWS, COLS)])
                second_tail = tf.schedule(
                    (second_staged, second_tail_dst), op=T.copy_async_tensor()
                )
                tails = tf.insert_slice(tails, second_tail, (TAIL + row, 0))
                return first, second, tails
