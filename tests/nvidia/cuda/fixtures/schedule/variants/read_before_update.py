"""Read the output through an identity call before the callee update."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class InplaceOutputUpdate:
    @func
    def identity(out: Tensor[(2, 4), "f32"]):
        return out

    @func
    def update(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                old = identity(out)
                row = tf.schedule((old[0:1, :], tf.empty(Tensor[(1, 4), "f32", (1, 4), "rmem"])), op=T.copy())
                return tf.insert_slice(old, row, (1, 0))

    @func
    def gemm(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            return update(out, x)  # noqa: F821 -- class-body binding
