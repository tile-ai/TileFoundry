"""A handoff in a loop on the first half of the threads, a read in a loop on the second half.

In the first half, one thread writes ``shared`` and another moves its head into
``tile`` each round. Both threads are in that half, so the barriers between them
run over it alone. Only the 64 thread mesh holds the mover and the readers, so
the barrier between the halves runs there.
"""
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="gemm", target=cuda.Target("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 64)))
class NestedSubsets:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(64,), names=("t",)) as threads:
                with threads[:32] as _producer:
                    shared = tf.empty(Tensor[(8,), "f32", (8,), "smem"])
                    tile = tf.empty(Tensor[(4,), "f32", (4,), "smem"])
                    for _row in range(2):
                        with threads[0:1] as _writer:
                            shared = tf.schedule((x, shared), op=T.copy())
                        with threads[1:2] as _mover:
                            tile = tf.schedule((shared[:4], tile), op=T.copy())
                with threads[32:] as _consumer:
                    out = tf.empty(Tensor[(4,), "f32"])
                    for _row in range(2):
                        out = tf.schedule((tile, out), op=T.copy())
                    return out
