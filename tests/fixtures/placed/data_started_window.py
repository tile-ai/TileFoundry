"""Minimal: a CTA reads a window whose start is a value loaded from a tensor."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

N, W, D, C = 64, 8, 16, 4


@module(
    entry="probe",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", C),),
)
class DataStart:
    @func
    def probe(x: Tensor[(N, D), "bf16"], starts: Tensor[(C,), "i64"]) -> Tensor[(C, W, D), "bf16"]:
        with Mesh(("cta",), layout=(C,), names=("c",)) as cta:
            start = tf.reshard(tf.reshape(starts[cta.c : cta.c + 1], new_shape=()), ((), (), {}), "rmem")
            tile = tf.reshard(x[start : start + W, :], ((W, D), (D, 1), {}), "smem")
            out = tf.zeros(Tensor[(C, W, D), "bf16"])
            back = tf.reshard(tf.reshape(tile, new_shape=(1, W, D)), ((1, W, D), (D * W, D, 1), {}), "gmem")
            return tf.insert_slice(out, back, (cta.c, 0, 0))


@module(
    entry="probe",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", C),),
)
class MeshStart:
    @func
    def probe(x: Tensor[(N, D), "bf16"], starts: Tensor[(C,), "i64"]) -> Tensor[(C, W, D), "bf16"]:
        with Mesh(("cta",), layout=(C,), names=("c",)) as cta:
            tile = tf.reshard(x[cta.c * W : cta.c * W + W, :], ((W, D), (D, 1), {}), "smem")
            out = tf.zeros(Tensor[(C, W, D), "bf16"])
            back = tf.reshard(tf.reshape(tile, new_shape=(1, W, D)), ((1, W, D), (D * W, D, 1), {}), "gmem")
            return tf.insert_slice(out, back, (cta.c, 0, 0))
