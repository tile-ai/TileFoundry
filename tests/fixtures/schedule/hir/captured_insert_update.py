"""Copy a scheduled row captured by a nested scope into its carried tile.

The indexed copy is lowered when the scope captures row. InsertSlice must
copy that existing result into the destination window instead of treating
it as an instruction that can still be redirected to the window.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.types import Layout
from tilefoundry.target import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class CapturedInsertUpdate:
    @func
    def gemm(
        x: Tensor[(4, 4), "bf16"], idx: Tensor[(2,), "i64"]
    ) -> Tensor[(2, 4), "bf16", "smem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                tile = tf.zeros(Tensor[(2, 4), "bf16", Layout((2, 4), (4, 1)), "smem"])
                for r in range(2):
                    row = tf.schedule(
                        (x, idx[r : r + 1]),
                        op=T.copy_async(smem_layout=Layout((1, 4), (4, 1)), fill=0),
                    )
                    with threads[:] as _carry:
                        tile = tf.insert_slice(tile, row, (r, 0))
                result = tile
        return result
