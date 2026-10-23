"""Give a carry initialized from a function parameter its own writable buffer.

The persistent input remains read-only: HIR explicitly copies it into acc before
entering the loop, then inserts each row into acc.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry_cuda import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class LoopCarryFromParam:
    @func
    def gemm(x: Tensor[(2, 4), "bf16"], y: Tensor[(1, 4), "bf16"]) -> Tensor[(2, 4), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                acc = tf.empty(Tensor[(2, 4), "bf16"])
                acc = tf.schedule((x, acc), op=T.copy())
                for r in range(2):
                    acc = tf.insert_slice(acc, y, (r, 0))
                result = acc
        return result
