"""A write in a loop on the first half of the threads, a read in a loop on the second half.

Only the 64 thread mesh holds both, so the barrier between them runs there; the
writer's loop needs its own barrier over its half alone.
"""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 64)))
class NestedSubsets:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(64,), names=("t",)) as threads:
                with threads[:32] as _producer:
                    shared = tf.empty(Tensor[(8,), "f32", (8,), "smem"])
                    for _row in range(2):
                        shared = tf.schedule((x, shared), op=T.copy())
                    tile = shared[:4]
                with threads[32:] as _consumer:
                    out = tf.empty(Tensor[(4,), "f32"])
                    for _row in range(2):
                        out = tf.schedule((tile, out), op=T.copy())
                    return out
