"""Split a copy into shared memory over a 16-thread mesh that all 32 threads run.

The mesh is not a slice of the thread level, so lowering guards nothing and
every thread of the CTA issues the copy. The split gives each row of the stage
to one of 16 positions, but 32 threads take those positions: two threads write
each row in one issue, and the sync family refuses it. Slicing the thread level
instead, as ``sm89_e4m3_gemm`` does, keeps each row to one thread.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

ROWS = 16
COLS = 8


@module(
    entry="gemm",
    target=cuda.Target("nvidia.rtx_6000_ada"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class SplitCopyFewerThreads:
    @func
    def gemm(x: Tensor[(ROWS, COLS), "f32"]) -> Tensor[(ROWS, COLS), "f32", "smem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(ROWS,), names=("row",)) as rows:
                split = ShardLayout(layout=Layout((ROWS, COLS), (COLS, 1)), attrs=(S(0),), mesh=rows)
                stage = tf.empty(Tensor[(ROWS, COLS), "f32", split, "smem"])
                staged = tf.schedule((tf.reshard(x, split), stage), op=T.copy())
                return tf.reshard(staged, Layout((ROWS, COLS), (COLS, 1)))
