"""A name bound before a mesh scope, rebound inside it, and returned after it."""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

N = 8


@module(entry="probe", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", N),))
class EscapedRebindInLoop:
    @func
    def probe(x: Tensor[(N, 4), "f32"]) -> Tensor[(N, 4), "f32"]:
        out = tf.zeros(Tensor[(N, 4), "f32"])
        with Mesh(("cta",), layout=(N,), names=("c",)) as cta:
            for i in range(1):
                t = cta.c + i
                row = tf.reshard(x[t : t + 1, :], (1, 4), "rmem")
                out = tf.insert_slice(out, tf.reshard(row * 2.0, (1, 4), "gmem"), (t, 0))
        return out
