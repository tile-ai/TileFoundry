"""Placed parallel-attention exchange across participant shards.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/parallel_attention/parallel_wrapper.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.all_to_all' (1 positional, keywords ['split_axis', 'concat_axis'])
ledger: OP-10

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
CTA_COUNT, THREAD_COUNT = BATCH * Q_HEADS, 4 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AllToAllModule1:
    """FlashInfer all_to_all entry."""

    @func
    def run(x: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, 4, 32),
            names=("rank", "head", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (BATCH @ mesh.rank, QUERY, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane), "rmem"
            )
            x_smem = tf.reshard(
                x_reg, (BATCH @ mesh.rank, QUERY, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane), "smem"
            )
            return tf.all_to_all(x_smem, split_axis=2, concat_axis=1)
