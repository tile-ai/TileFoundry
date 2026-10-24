"""Capture storage written by a two-CTA region in a later one-CTA region."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2), Topology("thread", 32)))
class MeshCaptureOutside:
    @func
    def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("block",)) as _first:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _writer:
                shared = tf.schedule((x, tf.empty(Tensor[(32,), "f32"])), op=T.copy())
        with Mesh(("cta",), layout=(1,), names=("block",)) as _second:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _reader:
                return tf.schedule((shared, tf.empty(Tensor[(32,), "f32"])), op=T.copy())
