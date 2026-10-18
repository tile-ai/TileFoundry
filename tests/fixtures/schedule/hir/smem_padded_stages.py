"""A staged slot reserves its padded address span and its own alignment."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class PaddedStages:
    @func
    def gemm(x: Tensor[(2, 3), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _block:
            with Mesh(("thread",), layout=(1,), names=("lane",)) as _lane:
                out = tf.zeros(Tensor[(2, 3), "f32"])
                for k in range(2):
                    tile = tf.schedule((x, tf.empty(Tensor[(2, 3), "f32", ((2, 3), (8, 1)), "smem"], buffers=2)), op=T.copy())
                    out = tf.schedule((tile, out), op=T.copy())
                return out
