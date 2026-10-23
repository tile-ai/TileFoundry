"""Changing only global ownership and retaining a sharded layout are views."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2), Topology("thread", 1)))
class ReshardViews:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(2,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _threads:
                partitioned = tf.reshard(x, ((2 @ cta.b, 4), (4, 1)), "gmem")
                same = tf.reshard(partitioned, ((2 @ cta.b, 4), (4, 1)), "gmem")
                whole = tf.reshard(same, (8,), "gmem")
                return tf.schedule((whole, tf.empty(Tensor[(8,), "f32", (8,), "rmem"])), op=T.copy())
