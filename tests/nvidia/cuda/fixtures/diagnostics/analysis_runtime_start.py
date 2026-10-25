from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="read", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class RuntimeStart:
    @func
    def read(x: Tensor[(16,), "f32"], start: Tensor[(), "i64"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            result = tf.zeros(Tensor[(4,), "f32", (4,), "rmem"])
            for n in range(start, 4):
                result = tf.reshard(x[n * 4 : n * 4 + 4], (4,), "rmem")
            return result
