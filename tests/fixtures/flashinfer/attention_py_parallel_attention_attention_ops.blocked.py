"""Data-dependent attention planning and CTA work dispatch.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/parallel_attention/attention_ops.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.dynamic_cta_dispatch' (4 positional, no keywords)
ledger: OP-12

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
class AttentionOpManagerModule1:
    """FlashInfer AttentionOpManager entry."""

    @func
    def run(
        qo_indptr: Tensor[(BATCH + 1,), "i32"],
        kv_indptr: Tensor[(BATCH + 1,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, 4, 32),
            names=("request", "head", "warp", "lane"),
        ) as mesh:
            qo_reg = tf.reshard(qo_indptr, (BATCH + 1,), "rmem")
            kv_smem = tf.reshard(kv_indptr, (BATCH + 1,), "smem")
            return tf.dynamic_cta_dispatch(qo_reg, kv_smem, mesh.request, mesh.head)
