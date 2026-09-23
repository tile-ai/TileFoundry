"""Persistent GEMM with a one-dimensional grid-stride CTA schedule."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

M = 3840
N = 4224
K = 4096
BM = 64
BN = 64
BK = 32
GRID_M = M // BM
GRID_N = N // BN
NUM_TILES = GRID_M * GRID_N
NBLOCKS = 132


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", NBLOCKS),),
)
class PersistentGemmFlat:
    """Each of 132 CTAs walks the flattened output-tile space by grid stride."""

    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "f32"]:
        out = tf.zeros(Tensor[(M, N), "f32"])
        with Mesh(("cta",), layout=(NBLOCKS,), names=("i",)) as cta:
            for t in range(cta.i, NUM_TILES, NBLOCKS):
                mi = (t // GRID_N) * BM
                ni = (t % GRID_N) * BN
                acc = tf.zeros(Tensor[(BM, BN), "f32", (BM, BN), "rmem"])
                for ki in tile(K, BK):
                    lhs = tf.reshard(a[mi : mi + BM, ki], (BM, BK), "smem")
                    rhs = tf.reshard(b[ki, ni : ni + BN], (BK, BN), "smem")
                    product = tf.cast(tf.matmul(lhs, rhs), dtype="f32")
                    acc = acc + tf.reshard(product, (BM, BN), "rmem")
                out = tf.insert_slice(
                    out,
                    tf.reshard(acc, (BM, BN), "gmem"),
                    (mi, ni),
                )
            return out


__all__ = [
    "BK",
    "BM",
    "BN",
    "GRID_M",
    "GRID_N",
    "K",
    "M",
    "N",
    "NBLOCKS",
    "NUM_TILES",
    "PersistentGemmFlat",
]
