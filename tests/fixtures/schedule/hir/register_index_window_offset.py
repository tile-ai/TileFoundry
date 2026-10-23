"""Lower arithmetic on a scalar register index through a dynamic slice window."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 1)))
class DynWindowOffset:
    @func
    def gemm(x: Tensor[(64,), "f32"], starts: Tensor[(1,), "i64"]) -> Tensor[(8,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _t:
                held = tf.schedule((starts, tf.empty(Tensor[(1,), "i64", (1,), "rmem"])), op=T.copy())
                start = tf.reshape(held, new_shape=())
                window = x[start + 1 : start + 9]
                return tf.schedule((window, tf.empty(Tensor[(8,), "f32", (8,), "rmem"])), op=T.copy())
