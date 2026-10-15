"""Weights whose local views cannot be represented on logical tensor axes."""
from tilefoundry.dsl import *
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CudaTarget


def weight_module(size, layout, attrs, mesh):
    @module(entry="read", target=CudaTarget("nvidia.h200_sxm"), topologies=mesh.topologies)
    class Weight:
        @func
        def read(weight: ConstTensor[(size,), "f32", ShardLayout(layout, attrs, mesh)]):
            return weight

    @runtime_module(Weight)
    class WeightTwin:
        @runtime_func
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
