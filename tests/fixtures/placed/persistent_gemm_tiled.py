"""Persistent GEMM with a two-dimensional rectangular CTA schedule."""

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
BX = 12
BY = 11
CHUNK_M = M // BX
CHUNK_N = N // BY


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", BX * BY),),
)
class PersistentGemmTiled:
    """Each of 132 CTAs owns one rectangle of output tiles."""

    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "f32"]:
        out = tf.zeros(Tensor[(M, N), "f32"])
        with Mesh(("cta",), layout=(BX, BY), names=("x", "y")) as cta:
            for mi in tile(cta.x * CHUNK_M, (cta.x + 1) * CHUNK_M, BM):
                for ni in tile(cta.y * CHUNK_N, (cta.y + 1) * CHUNK_N, BN):
                    acc = tf.zeros(Tensor[(BM, BN), "f32", (BM, BN), "rmem"])
                    for ki in tile(K, BK):
                        lhs = tf.reshard(a[mi, ki], (BM, BK), "smem")
                        rhs = tf.reshard(b[ki, ni], (BK, BN), "smem")
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
    "BX",
    "BY",
    "CHUNK_M",
    "CHUNK_N",
    "K",
    "M",
    "N",
    "PersistentGemmTiled",
]
