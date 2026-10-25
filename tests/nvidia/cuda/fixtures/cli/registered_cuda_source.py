import json

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_extent = json.loads('{"extent": 1}')["extent"]


@module(
    entry="model",
    target=CudaTarget("vendor.v100_sxm2_32gb"),
    topologies=(Topology("cta", _extent),),
)
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
