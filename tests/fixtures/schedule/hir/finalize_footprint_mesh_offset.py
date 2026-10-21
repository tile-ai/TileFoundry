"""Print a loop footprint reading a previous launch at mesh-coordinate offsets."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

N = 4
R = 32


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", N), Topology("thread", R)),
)
class FootprintMeshCoord:
    @func
    def gemm(x: Tensor[(N * R,), "f32"]):
        with Mesh(("cta",), layout=(N,), names=("m",)) as cta:
            out = tf.empty(Tensor[(N * R,), "f32"])
            with Mesh(("thread",), layout=(R,), names=("row",)) as rows:
                r0 = cta.m * R
                t = tf.schedule(
                    (x[r0 : r0 + R], tf.empty(Tensor[(R,), "f32", ((R @ rows.row,), (1,)), "rmem"])),
                    op=T.copy(),
                )
                y = tf.schedule((t + t, tf.empty(Tensor[(R,), "f32", (R,)])), op=T.copy())
                out = tf.insert_slice(out, y, (r0,))
        with Mesh(("cta",), layout=(N,), names=("m",)) as cta2:
            res = tf.empty(Tensor[(N * R,), "f32"])
            with Mesh(("thread",), layout=(R,), names=("row",)) as rows2:
                acc = tf.zeros(Tensor[(R,), "f32", ((R @ rows2.row,), (1,)), "rmem"])
                for n in range(N):
                    u = tf.schedule(
                        (out[n * R : n * R + R], tf.empty(Tensor[(R,), "f32", ((R @ rows2.row,), (1,)), "rmem"])),
                        op=T.copy(),
                    )
                    acc = acc + u
                z = tf.schedule((acc, tf.empty(Tensor[(R,), "f32", (R,)])), op=T.copy())
                res = tf.insert_slice(res, z, (cta2.m * R,))
        return res
