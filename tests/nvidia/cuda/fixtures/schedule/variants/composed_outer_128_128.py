from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(128,256), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:  # noqa: F841
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:  # noqa: F841
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout((128,(4,64)), (64,(8192,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(128,256), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,128:128+128], (16, 8, 8, 16))
                return tf.schedule((reshaped, tf.empty(Tensor[(16, 8, 8, 16), "bf16", "rmem"])), op=T.copy())
