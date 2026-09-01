"""Placed indexed attention composed from HIR primitives.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/msa_ops/sparse_prefill.py
license: Apache-2.0 (no upstream source is vendored)

Page/block gather, score normalization, and value reduction follow the
primitive structure in tests/fixtures/placed/gqa_decode.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * Q_HEADS, 8 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class MsaSparseAttentionModule1:
    """FlashInfer msa_sparse_attention entry.

    predicted-ns: 196255608
    waves: 1
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def run(
        q: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        v: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        block_indices: Tensor[(BATCH, QUERY, 16), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, 8, 32),
            names=("batch", "head", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, (BATCH @ mesh.batch, QUERY, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_gmem = tf.reshard(
                k, (BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "gmem"
            )
            v_gmem = tf.reshard(
                v, (BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "gmem"
            )
            index_reg = tf.reshard(block_indices, (BATCH @ mesh.batch, QUERY, 16), "rmem")
            blocked_k = tf.reshape(
                k_gmem,
                new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
            )
            blocked_v = tf.reshape(
                v_gmem,
                new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
            )
            output = tf.zeros(Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"])

            for query_index in range(QUERY):
                query_blocks = tf.slice(
                    index_reg,
                    (0, query_index, 0),
                    sizes=(BATCH, 1, 16),
                    strides=(1, 1, 1),
                )
                flat_blocks = tf.reshape(query_blocks, new_shape=(BATCH * 16,))
                selected_k = tf.reshape(
                    tf.index_select(blocked_k, flat_blocks, dim=0),
                    new_shape=(BATCH, 1, 16 * 16, KV_HEADS, HEAD_DIM),
                )
                selected_v = tf.reshape(
                    tf.index_select(blocked_v, flat_blocks, dim=0),
                    new_shape=(BATCH, 1, 16 * 16, KV_HEADS, HEAD_DIM),
                )
                selected_k = tf.repeat_interleave(
                    selected_k, repeats=Q_HEADS // KV_HEADS, axis=3
                )
                selected_v = tf.repeat_interleave(
                    selected_v, repeats=Q_HEADS // KV_HEADS, axis=3
                )
                selected_k = tf.reshard(
                    selected_k,
                    (BATCH @ mesh.batch, 1, 16 * 16, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "rmem",
                )
                selected_v = tf.reshard(
                    selected_v,
                    (BATCH @ mesh.batch, 1, 16 * 16, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "rmem",
                )
                query = tf.slice(
                    q_reg,
                    (0, query_index, 0, 0),
                    sizes=(BATCH, 1, Q_HEADS, HEAD_DIM),
                    strides=(1, 1, 1, 1),
                )
                query = tf.reshape(
                    tf.cast(query, "f32"),
                    new_shape=(BATCH, 1, 1, Q_HEADS, HEAD_DIM),
                )
                key = tf.cast(selected_k, "f32")
                value = tf.cast(selected_v, "f32")
                raw = tf.reduce(query * key, axes=(-1,), keepdim=True, kind="sum")
                score = raw * tf.full_like(raw, value=0.08838834764831845)
                peak = tf.reduce(score, axes=(-3,), keepdim=True, kind="max")
                weight = tf.exp(score - peak)
                total = tf.reduce(weight, axes=(-3,), keepdim=False, kind="sum")
                blended = tf.reduce(weight * value, axes=(-3,), keepdim=False, kind="sum")
                attended = tf.cast(blended / total, "bf16")
                attended = tf.reshard(
                    attended,
                    (BATCH @ mesh.batch, 1, Q_HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "gmem",
                )
                output = tf.insert_slice(output, attended, (0, query_index, 0, 0))

            return output
