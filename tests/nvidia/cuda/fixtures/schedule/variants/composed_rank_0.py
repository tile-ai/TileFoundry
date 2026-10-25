from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(64,128), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:  # noqa: F841
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:  # noqa: F841
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout(((8,8),(2,4,16)), ((512,64),(4096,16,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(64,128), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,0:0+64], (8,8,16,4))
                return tf.schedule((reshaped, tf.empty(Tensor[(8,8,16,4), "bf16", "rmem"])), op=T.copy())
