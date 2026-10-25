"""Return the output beside the callee update, reached through a relay."""

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
                row = tf.schedule((x, tf.empty(Tensor[(1, 4), "f32", (1, 4), "rmem"])), op=T.copy())
                return tf.insert_slice(out, row, (1, 0))

    @func
    def relay(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        return update(out, x)

    @func
    def gemm(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            return relay(out, x), out  # noqa: F821 -- class-body binding
