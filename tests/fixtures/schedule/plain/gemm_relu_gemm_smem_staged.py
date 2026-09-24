"""The matmul-ReLU-matmul of `gemm_relu_gemm_tiled`, with both operand windows staged.

Same computation, same tiling, same accumulator: the one difference is that
each k step names the shared-memory tile it contracts out of instead of letting
the contraction read the global window straight. Staging is a `reshard` and the
instruction that performs it is an offer --- `T.copy_async`, `T.tma` or a plain
`T.copy` --- so what used to be a second implementation of the contraction is a
decision of its own, and ReLU stays the first matmul's tile-level epilogue.
"""
from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 1024
N = 2048
K = 2048
BM = 32
BN = 16
BK = 32


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 512)))
class GEMM_RELU_GEMM_SMEM_STAGED:
    @func
    def gemm(a: Tensor[(M, K), "bf16"],
             b: Tensor[(K, N), "bf16"],
             c: Tensor[(N, N), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            first = tf.zeros(Tensor[(M, N), "bf16"])
            for m in tile(M, BM):
                for n in tile(N, BN):
                    acc = tf.zeros(Tensor[(BM, BN), "f32", (BM, BN), "rmem"])
                    for k in tile(K, BK):
                        a_s = tf.reshard(a[m, k], (BM, BK), "smem")
                        b_s = tf.reshard(b[k, n], (BK, BN), "smem")
                        lhs = tf.cast(a_s, dtype="f32")
                        rhs = tf.cast(b_s, dtype="f32")
                        partial = tf.reshard(
                            tf.matmul(lhs, rhs), (BM, BN), "rmem"
                        )
                        acc = acc + partial
                    first = tf.insert_slice(first, tf.relu(tf.cast(acc, dtype="bf16")), (m, n))

            result = tf.zeros(Tensor[(M, N), "bf16"])
            for m2 in tile(M, BM):
                for n2 in tile(N, BN):
                    acc2 = tf.zeros(Tensor[(BM, BN), "f32", (BM, BN), "rmem"])
                    for k2 in tile(N, BK):
                        first_s = tf.reshard(first[m2, k2], (BM, BK), "smem")
                        c_s = tf.reshard(c[k2, n2], (BK, BN), "smem")
                        lhs2 = tf.cast(first_s, dtype="f32")
                        rhs2 = tf.cast(c_s, dtype="f32")
                        partial2 = tf.reshard(
                            tf.matmul(lhs2, rhs2), (BM, BN), "rmem"
                        )
                        acc2 = acc2 + partial2
                    result = tf.insert_slice(result, tf.cast(acc2, dtype="bf16"), (m2, n2))
            return result
