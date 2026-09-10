"""Placed A03 logits soft-cap attention boundary.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/attention/variants.cuh:DefaultAttention
license: Apache-2.0 (no upstream source is vendored)

CTAs own query rows while four warps and 32 lanes vectorize each head. This
matches the upstream attention variant's row-parallel, vectorized score path.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, WIDTH = 256, 128
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


@module(entry="softcap_attention", target=TARGET, topologies=TOPOLOGIES)
class SoftcapAttention:
    """FlashInfer DefaultAttention logits soft-cap kernel.

    predicted-ns: 14814  waves: 2
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def softcap_attention(
        q: Tensor[(ROWS, WIDTH), "f32"],
        k: Tensor[(ROWS, WIDTH), "f32"],
        v: Tensor[(ROWS, WIDTH), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("query", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (ROWS @ mesh.query, 4 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            k_reg = tf.reshard(
                k,
                (ROWS, 4 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            v_reg = tf.reshard(
                v,
                (ROWS, 4 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            q_smem = tf.reshard(
                q_reg,
                (ROWS @ mesh.query, 4 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            k_smem = tf.reshard(
                k_reg,
                (ROWS, 4 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            v_smem = tf.reshard(
                v_reg,
                (ROWS, 4 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            scores = q_smem @ tf.transpose(k_smem, perm=(1, 0))
            reduced_scores = tf.reshard(
                scores,
                (ROWS @ mesh.query, ROWS),
                "smem",
            )
            probabilities = tf.softmax(8.0 * tf.tanh(reduced_scores / 8.0), axis=-1)
            output = probabilities @ v_smem
            return tf.reshard(
                output,
                (ROWS @ mesh.query, 4 @ mesh.warp, 32 @ mesh.lane),
                "gmem",
            )


__all__ = ["SoftcapAttention"]
