"""Packed block-scaled attention payload and scale production.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/attention/cute_dsl/fmha_blockscaled.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: REG-05, OP-01

Each SURVEY entry remains an independent Module with the source group's
representative supported specialization and upstream mechanism intact.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * Q_HEADS * (QUERY // 16), 8 * 16
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class CuteDslFmhaBlockscaledPrefillModule1:
    """FlashInfer cute_dsl_fmha_blockscaled_prefill entry."""

    @func
    def run(x: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, QUERY // 16, 8, 16),
            names=("batch", "head", "token_block", "token", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            packed, scales = tf.quant(tf.cast(x_smem, "f32"), group=16, target_dtype="nvfp4")
            return packed, scales
