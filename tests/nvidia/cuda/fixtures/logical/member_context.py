import tilefoundry
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types import Topology
from tilefoundry_nvidia import CudaTarget


@tilefoundry.module(
    entry="attention",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class Layer:
    @tilefoundry.func
    def attention(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), (2,), ("block",)) as _cta:
            return tf.relu(x)

    @tilefoundry.func(topologies=(Topology("thread", 4),))
    def softmax(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("thread",), (4,), ("lane",)) as _thread:
            return tf.square(x)
