"""A matmul followed by an RMS norm, repeated in a child Module.

Selectors, dimension parsing, topology rejection, and report rendering consume
this program. It deliberately carries no Mesh, Reshard, or ShardLayout.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class CMine:
    @func
    def root(
        x: Tensor[(16, 16), "bf16"],
        w: Tensor[(16, 16), "bf16"],
        weight: Tensor[(16,), "f32"],
    ) -> Tensor[(16, 16), "bf16"]:
        h = tf.matmul(x, w)
        return tf.rms_norm(h, weight)

    @module(entry="inner")
    class child:
        @func
        def inner(
            x: Tensor[(16, 16), "bf16"],
            w: Tensor[(16, 16), "bf16"],
            weight: Tensor[(16,), "f32"],
        ) -> Tensor[(16, 16), "bf16"]:
            h = tf.matmul(x, w)
            return tf.rms_norm(h, weight)
