"""A shared slice produced in one thread submesh remains usable in its sibling."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 64)))
class SubmeshSliceInSibling:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(64,), names=("t",)) as threads:
                with threads[:32] as _producer:
                    shared = tf.schedule((x, tf.empty(Tensor[(8,), "f32", (8,), "smem"])), op=T.copy())
                    tile = shared[:4]
                with threads[32:] as _consumer:
                    return tf.schedule((tile, tf.empty(Tensor[(4,), "f32"])), op=T.copy())
