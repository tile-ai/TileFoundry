from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

N = DimVar("N", 1, 8)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class Open:
    @func
    def main(x: Tensor[(N,), "f32"]):
        return tf.add(x, x)
