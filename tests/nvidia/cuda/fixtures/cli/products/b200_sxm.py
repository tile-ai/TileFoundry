from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="model", target=CudaTarget("nvidia.b200_sxm"), topologies=(Topology("cta", 1),))
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
