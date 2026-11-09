from tilefoundry import target
from tilefoundry.dsl import *

_target = target.registered()["vendor.npu"]()


@module(entry="model", target=_target, topologies=(Topology("core", 1),))
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
