from tilefoundry.dsl import *
from tilefoundry.target import registered_targets

_target = registered_targets()["vendor.npu"]()


@module(entry="model", target=_target, topologies=(Topology("core", 1),))
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
