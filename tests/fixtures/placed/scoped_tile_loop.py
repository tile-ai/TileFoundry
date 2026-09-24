"""One CTA's share of the rows, taken by the threads inside the scope it runs in.

The scope in force names both levels, so the region inside the loop names only
the threads: a suffix of those levels, refining them rather than starting
again. The loop starts at this CTA's own row and runs to the end, so how often
it runs is read at its widest rather than one coordinate at a time, and the
value it carries is bound inside the thread region it is read in.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

CTAS, THREADS = 4, 128
ROWS, COLS, BLK = 64, 32, 8
_H200 = CudaTarget("nvidia.h200_sxm")


@module(
    entry="row_relu",
    target=_H200,
    topologies=(Topology("cta", CTAS), Topology("thread", THREADS)),
)
class ScopedTileLoop:
    @func
    def row_relu(x: Tensor[(ROWS, COLS), "f32"]) -> Tensor[(ROWS, COLS), "f32"]:
        out = tf.zeros(Tensor[(ROWS, COLS), "f32"])
        with Mesh(("cta", "thread"), layout=(CTAS, THREADS), names=("c", "t")) as whole:
            for r in tile(whole.c * BLK, ROWS, BLK):
                with Mesh(("thread",), layout=(THREADS,), names=("t",)) as _inside:
                    out = tf.insert_slice(out, tf.relu(x[r, :]), (r, 0))
            return out
