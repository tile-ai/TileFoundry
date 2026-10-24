"""An E4M3 GEMM for one sm_89 warp as plain HIR.

A (16, 128) times B, the (K, N) view of a row-major (N, K) weight, into F32. K
runs in four 32-wide slices; each slice is loaded into registers, where the SM89
warp MMA reads its operands, and its product is added into the F32 accumulator.
"""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

M = 16
N = 8
K = 128
BK = 32


@module(entry="gemm", target=CudaTarget("nvidia.rtx_6000_ada"),
        topologies=(Topology("cta", 1), Topology("thread", 32)))
class SM89_E4M3_GEMM:
    @func
    def gemm(a: Tensor[(M, K), "fp8e4m3"],
             b: Tensor[(K, N), "fp8e4m3", ((K, N), (1, K))]) -> Tensor[(M, N), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            acc = tf.zeros(Tensor[(M, N), "f32", "rmem"])
            for kb in range(K // BK):
                at = tf.reshard(a[:, kb * BK:kb * BK + BK], (M, BK), "rmem")
                bt = tf.reshard(b[kb * BK:kb * BK + BK, :], (BK, N), "rmem")
                acc = acc + tf.matmul(at, bt, out_dtype="f32")
            return tf.reshard(acc, (M, N), "gmem")
