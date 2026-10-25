"""Return the output parameter unchanged from a callee, and use it again in the caller."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class InplaceOutputUpdate:
    @func
    def update(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                row = tf.schedule(  # noqa: F841
                    (x, tf.empty(Tensor[(1, 4), "f32", (1, 4), "rmem"])), op=T.copy()
                )
                return out

    @func
    def gemm(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            return update(out, x), out  # noqa: F821 -- class-body binding
