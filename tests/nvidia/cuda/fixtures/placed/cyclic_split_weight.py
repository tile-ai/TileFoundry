"""Weights whose local piece is not one block of the logical tensor."""
from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


def weight_module(size, layout, attrs, mesh):
    @module(entry="read", target=cuda.Target("nvidia.h200_sxm"), topologies=(*mesh.topologies, Topology("cta", 1)))
    class Weight:
        @func
        def read(weight: ConstTensor[(size,), "f32", ShardLayout(layout, attrs, mesh)]):
            return tf.reshard(weight)

    @runtime.module(Weight)
    class WeightTwin:
        @runtime.func
        def read(self, weight):
            return weight

    return Weight, WeightTwin


GPU4 = Topology("gpu", 4)
GPU32 = Topology("gpu", 32)
CyclicSplitWeight, CyclicSplitWeightTwin = weight_module(
    8, Layout((2, 4), (4, 1)), (Split(1),), Mesh((GPU4,), Layout((4,), (1,)), ("g",)),
)
MeshOrderWeight, MeshOrderWeightTwin = weight_module(
    32, Layout((4, 8), (8, 1)), (Split(1), Split(0)),
    Mesh((GPU32,), Layout((8, 4), (4, 1)), ("g", "h")),
)
