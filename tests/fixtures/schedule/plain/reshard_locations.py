"""Reshards distinguish changed addresses, owners, and views."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2), Topology("thread", 2)))
class ReshardLocations:
    @func
    def gemm(x: Tensor[(8, 8), "f32"], y: Tensor[(8,), "f32"]):
        with Mesh(("cta",), layout=(2,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(2,), names=("t",)) as threads:
                shared = tf.reshard(x, Layout((8, 8), (8, 1)), "smem")
                transposed = tf.reshard(shared, Layout((8, 8), (1, 8)), "smem")
                held = tf.reshard(y, ((2 @ threads.t, 4), (4, 1)), "rmem")
                exchanged = tf.reshard(held, ((2, 2 @ threads.t, 2), (4, 2, 1)), "rmem")
                repartitioned = tf.reshard(y, ((2 @ cta.b, 4), (4, 1)), "gmem")
                same = tf.reshard(repartitioned, ((2 @ cta.b, 4), (4, 1)), "gmem")
                shared_view = tf.reshard(shared, ((2 @ threads.t, 4, 8), (32, 8, 1)), "smem")
                placed = tf.reshard(x, ((2 @ cta.b, 4, 8), (32, 8, 1)), "gmem")
                window = placed[:, 1:8]
                shifted = tf.reshard(window, ((2 @ cta.b, 4, 7), (32, 8, 1)), "gmem")
                return transposed, exchanged, repartitioned, same, shared_view, shifted
