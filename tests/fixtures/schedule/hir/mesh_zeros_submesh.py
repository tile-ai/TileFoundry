"""Fill shared storage only in the submesh containing its zeros operation."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)))
class MeshZerosSubmesh:
    @func
    def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                with threads[0:16] as _half:
                    tile = tf.zeros(Tensor[(32,), "f32", Layout((32,), (1,)), "smem"])
                return tf.schedule((tile, tf.empty(Tensor[(32,), "f32"])), op=T.copy())
