from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

N = 132 * 128
_H200 = CudaTarget("nvidia.h200_sxm")


@module(entry="nope", target=_H200, topologies=(Topology("cta", 132),))
class Unsound:
    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        return tf.square(x)


@module(entry="kernel", target=_H200, topologies=(Topology("cta", 132),))
class Sound:
    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("cta",), layout=(132,), names=("block",)) as m:
            placed = tf.reshard(x, (N @ m.block,), "gmem")
            return tf.reshard(tf.square(placed), (N @ m.block,), "gmem")
