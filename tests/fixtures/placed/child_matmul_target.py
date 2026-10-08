"""Child Modules with no Target of their own, reached from a CUDA root.

Each child carries an on-chip matmul result around its K loop. Only a root may
declare a Target, so a child's is the root's, and the carried value is where a
CUDA MMA writes its result: an f32 register tile. ``ChildMatmulDirect`` hands
that tile straight back, so the root's own call is typed by the same rule;
``ChildMatmulStaged`` keeps the loop in a specialization variant.
"""

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Mesh, RangePattern, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
BK = 16
K_LEN = DimVar("k_len", BK, K)


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


@module(entry="direct", topologies=(Topology("cta", 1),))
class ChildMatmulDirect:
    @func
    def direct(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            lhs = tf.reshard(a[:, 0:BK], (M, BK), "smem")
            rhs = tf.reshard(b[0:BK, :], (BK, N), "smem")
            acc = tf.matmul(lhs, rhs, out_dtype="f32")
            for k in tile(BK, K, BK):
                lhs = tf.reshard(a[:, k], (M, BK), "smem")
                rhs = tf.reshard(b[k, :], (BK, N), "smem")
                acc = acc + tf.matmul(lhs, rhs, out_dtype="f32")
            return acc


@module(entry="staged", topologies=(Topology("cta", 1),))
class ChildMatmulStaged:
    @func
    def staged(
        a: Tensor[(M, K_LEN), "bf16"], b: Tensor[(K_LEN, N), "bf16"]
    ) -> Tensor[(M, N), "f32"]:
        pass

    @staged.specialize(RangePattern("k_len", BK, K))
    def staged_any_length(
        a: Tensor[(M, K_LEN), "bf16"], b: Tensor[(K_LEN, N), "bf16"]
    ) -> Tensor[(M, N), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            lhs = tf.reshard(a[:, 0:BK], (M, BK), "smem")
            rhs = tf.reshard(b[0:BK, :], (BK, N), "smem")
            acc = tf.matmul(lhs, rhs, out_dtype="f32")
            for k in tile(BK, K_LEN, BK):
                lhs = tf.reshard(a[:, k], (M, BK), "smem")
                rhs = tf.reshard(b[k, :], (BK, N), "smem")
                acc = acc + tf.matmul(lhs, rhs, out_dtype="f32")
            return tf.reshard(acc, (M, N), "gmem")


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class ChildMatmulRoot:
    child = ChildMatmul
    direct = ChildMatmulDirect
    staged = ChildMatmulStaged

    @func
    def gemm(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "f32"]:
        return child(a, b)  # noqa: F821 -- class-body binding

    @func
    def gemm_on_chip(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
        return direct(a, b)  # noqa: F821 -- class-body binding

    @func
    def gemm_staged(
        a: Tensor[(M, K_LEN), "bf16"], b: Tensor[(K_LEN, N), "bf16"]
    ) -> Tensor[(M, N), "f32"]:
        return staged(a, b)  # noqa: F821 -- class-body binding
