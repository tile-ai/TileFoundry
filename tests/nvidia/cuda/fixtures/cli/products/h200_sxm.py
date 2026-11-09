from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="model", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
