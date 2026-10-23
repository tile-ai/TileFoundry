from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Tensor, Topology, tf
from tilefoundry_cuda import CudaTarget

T = DimVar("T", 1, 4096)
C = DimVar("C", 1, 4096)

@module(entry="squeeze", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class TwoDimVarReshape:
    @func
    def squeeze(x: Tensor[(1, T, 1, C), "f32"]) -> Tensor[(T, C), "f32"]:
        return tf.reshape(x, (T, C))
