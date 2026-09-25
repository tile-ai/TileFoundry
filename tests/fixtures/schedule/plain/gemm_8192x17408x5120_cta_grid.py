"""The real matmul as plain HIR: a CTA grid, and no instruction chosen yet.

M = 8192, K = 5120, N = 17408, split over a 64 x 68 grid of CTAs, one
128 x 256 tile of the output each (``M @ cta.bm``, ``N @ cta.bn``), with K
streamed through shared memory 64 at a time.  Nothing here is scheduled and
nothing names a thread: this is the program an author asks ``schedule
candidates`` about before writing the schedule, so what it states is each
CTA's tiles and the storage they move between.
"""
from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
GM = M // BM
GN = N // BN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", GM * GN), Topology("thread", 384)))
class GEMM_8192X17408X5120_CTA_GRID:
    @func
    def gemm(a: Tensor[(M, K), "bf16"],
             b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GM, GN), names=("bm", "bn")) as cta:
            acc = tf.zeros(Tensor[(M @ cta.bm, N @ cta.bn), "f32", "rmem"])
            for k in tile(K, BK):
                at = tf.reshard(a[:, k], (M @ cta.bm, BK), "smem")
                bt = tf.reshard(b[k, :], (BK, N @ cta.bn), "smem")
                part = tf.matmul(at, bt)
                acc = acc + tf.reshard(tf.cast(part, "f32"), (M @ cta.bm, N @ cta.bn), "rmem")
            return tf.reshard(tf.cast(acc, "bf16"), (M, N), "gmem")
