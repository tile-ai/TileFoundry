import json

from tilefoundry import target
from tilefoundry.dsl import *

_extent = json.loads('{"extent": 1}')["extent"]
_target = target.registered()["vendor.npu"]()


@module(entry="model", target=_target, topologies=(Topology("core", _extent),))
class Model:
    @func
    def model(source: Tensor[(8,), "f32"]):
        return tf.add(source, source)
