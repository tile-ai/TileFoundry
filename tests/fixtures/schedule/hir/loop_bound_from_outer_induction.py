"""Start an inner range with arithmetic on the outer induction variable."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

N = 8

R = 32

@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", R)))
class LoopStart:
    @func
    def gemm(x: Tensor[(2 * N * R,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("one",)) as _cta:
            with Mesh(("thread",), layout=(R,), names=("lane",)) as th:
                acc = tf.zeros(Tensor[(R,), "f32", ((R @ th.lane,), (1,)), "rmem"])
                for t in range(N):
                    for n in range((t * 3) // 2, 2 * N):
                        u = tf.schedule((x[n * R : n * R + R], tf.empty(Tensor[(R,), "f32", ((R @ th.lane,), (1,)), "rmem"])), op=T.copy())
                        acc = acc + u
                return tf.schedule((acc, tf.empty(Tensor[(R,), "f32", (R,)])), op=T.copy())
