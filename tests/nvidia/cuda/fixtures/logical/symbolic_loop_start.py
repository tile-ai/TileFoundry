from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

NC = DimVar("chunks", 1, 32)


@module(entry="read", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class SymbolicStart:
    @func
    def read(x: Tensor[(16,), "f32"]):
        with Mesh(("cta",), layout=(4,), names=("t",)) as _cta:
            result = tf.zeros(Tensor[(4,), "f32", (4,), "rmem"])
            for n in range(16 // NC, 4):
                result = tf.reshard(x[n * 4 : n * 4 + 4], (4,), "rmem")
            return result
