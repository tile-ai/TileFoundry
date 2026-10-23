"""Write a scheduled row captured by a nested scope into its carried tile.

Allocation binds the indexed copy to the tile's destination window. Lowering
uses that decision at the producer, before entering the capturing scope.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.types import Layout
from tilefoundry_cuda import CudaTarget


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
                    row_dst = tf.empty(
                        Tensor[
                            (1, 4), "bf16",
                            (1, 4),
                            "smem",
                        ],
                    )
                    row = tf.schedule(
                        (x, row_dst, idx[r : r + 1]),
                        op=T.copy_async(fill=0),
                    )
                    with threads[:] as _carry:
                        tile = tf.insert_slice(tile, row, (r, 0))
                result = tile
        return result
