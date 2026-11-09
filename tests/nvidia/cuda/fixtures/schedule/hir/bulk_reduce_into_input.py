"""Reduce two CTA partial sums into a global destination the caller zeroes and passes in."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="gemm",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 32)),
)
class BulkReduceIntoInput:
    @func
    def gemm(dq0: Tensor[(64, 64), "f32"]):
        with Mesh(("cta",), layout=(2,), names=("k",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                part_s = tf.zeros(Tensor[(64, 64), "f32", (64, 64) @ {cta.k @ P("sum")}, "smem"])
                return tf.schedule((part_s, dq0), op=T.copy_reduce_async_tensor())
