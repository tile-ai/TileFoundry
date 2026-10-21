"""Reduce two CTA partial sums from shared memory into an explicit global destination."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 32)))
class BulkReduceStore:
    @func
    def gemm():
        with Mesh(("cta",), layout=(2,), names=("k",)) as cta:
            part_s = tf.zeros(Tensor[(64, 64), "f32", (64, 64) @ {cta.k @ P("sum")}, "smem"])
            dq0 = tf.zeros(Tensor[(64, 64), "f32"])
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                return tf.schedule((part_s, dq0), op=T.copy_reduce_async_tensor())
