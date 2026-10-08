"""A child Module with no Target of its own, reached from a CUDA root.

The child carries an on-chip matmul result around its K loop. Only a root may
declare a Target, so the child's is the root's, and the carried value is where
a CUDA MMA writes its result: an f32 register tile.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
BK = 16


@module(entry="run", topologies=(Topology("cta", 1),))
class ChildMatmul:
    @func
    def run(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            lhs = tf.reshard(a[:, 0:BK], (M, BK), "smem")
            rhs = tf.reshard(b[0:BK, :], (BK, N), "smem")
            acc = tf.matmul(lhs, rhs, out_dtype="f32")
            for k in tile(BK, K, BK):
                lhs = tf.reshard(a[:, k], (M, BK), "smem")
                rhs = tf.reshard(b[k, :], (BK, N), "smem")
                acc = acc + tf.matmul(lhs, rhs, out_dtype="f32")
            return tf.reshard(acc, (M, N), "gmem")


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class ChildMatmulRoot:
    child = ChildMatmul

    @func
    def gemm(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "f32"]:
        return child(a, b)  # noqa: F821 -- class-body binding
