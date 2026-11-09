import tilefoundry
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir import types
from tilefoundry_nvidia import cuda


@tilefoundry.module(
    entry="attention",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(types.Topology("cta", 2),),
)
class Layer:
    @tilefoundry.func
    def attention(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), (2,), ("block",)) as _cta:
            return tf.relu(x)

    @tilefoundry.func(topologies=(types.Topology("thread", 4),))
    def softmax(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("thread",), (4,), ("lane",)) as _thread:
            return tf.square(x)
