"""Placed causal attention with one open dimension in each execution regime.

Decode places a one-token query while scanning an open context. Prefill places
an open query sequence, then scans every key block for each query block while
masking positions beyond the global causal boundary.
"""

from __future__ import annotations

import math

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, DimVarRangePat, Mesh, Tensor, ceildiv, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare tile() in authored bodies
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

SEQ = DimVar("seq", 1, 4097)
CTX = DimVar("ctx", 1, 4097)

HEADS = 16
HEAD_DIM = 64
BLOCK = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)


@module(
    entry="attend",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", HEADS),),
)
class PrefillDecodeAttention:
    @func
    def attend(
        q: Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"]:
        pass

    @attend.specialize(DimVarRangePat("seq", 1, 2))
    def decode(
        q: Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"]:
        with Mesh(("cta",), layout=(HEADS,), names=("head",)) as cta:
            padded_k = tf.insert_slice(
                tf.zeros(Tensor[(1, ceildiv(CTX, BLOCK) * BLOCK, HEADS, HEAD_DIM), "bf16"]),
                k_cache,
                (0, 0, 0, 0),
            )
            padded_v = tf.insert_slice(
                tf.zeros(Tensor[(1, ceildiv(CTX, BLOCK) * BLOCK, HEADS, HEAD_DIM), "bf16"]),
                v_cache,
                (0, 0, 0, 0),
            )
            context_size = tf.shape_of(k_cache)[1]
            query_start = context_size - 1
            qh = tf.reshard(q, (1, SEQ, HEADS @ cta.head, HEAD_DIM), "smem")
            qt = tf.transpose(tf.cast(qh, dtype="f32"), perm=(0, 2, 1, 3))
            qs = qt * tf.full_like(qt, value=SCALE)
            template = tf.reduce(qt, axes=(-1,), keepdim=True, kind="sum")
            running_max = tf.full_like(template, value=-1e30)
            running_sum = tf.full_like(template, value=0.0)
            running_out = tf.full_like(qt, value=0.0)

            for window in tile(ceildiv(CTX, BLOCK) * BLOCK, BLOCK):
                kb = tf.reshard(
                    padded_k[:, window, :, :],
                    (1, BLOCK, HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                vb = tf.reshard(
                    padded_v[:, window, :, :],
                    (1, BLOCK, HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                kt = tf.transpose(tf.cast(kb, dtype="f32"), perm=(0, 2, 3, 1))
                vt = tf.transpose(tf.cast(vb, dtype="f32"), perm=(0, 2, 1, 3))
                query_positions = (
                    tf.reshape(tf.arange(Tensor[(SEQ,), "i64", "gmem"]), new_shape=(SEQ, 1))
                    + query_start
                )
                key_positions = (
                    tf.reshape(tf.arange(Tensor[(BLOCK,), "i64", "gmem"]), new_shape=(1, BLOCK))
                    + window
                )
                keep = key_positions <= query_positions
                raw_scores = tf.matmul(qs, kt)
                scores = tf.where(
                    keep,
                    raw_scores,
                    tf.full_like(raw_scores, value=-1e30),
                )
                block_max = tf.reduce(scores, axes=(-1,), keepdim=True, kind="max")
                next_max = tf.max(running_max, block_max)
                correction = tf.exp(running_max - next_max)
                probabilities = tf.exp(scores - next_max)
                next_sum = running_sum * correction + tf.reduce(
                    probabilities, axes=(-1,), keepdim=True, kind="sum"
                )
                next_out = running_out * correction + tf.matmul(probabilities, vt)
                running_max = next_max
                running_sum = next_sum
                running_out = next_out

            normalized = running_out / running_sum
            return tf.transpose(tf.cast(normalized, dtype="bf16"), perm=(0, 2, 1, 3))

    @attend.specialize(DimVarRangePat("seq", 2, 4097))
    def prefill(
        q: Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, SEQ, HEADS, HEAD_DIM), "bf16"]:
        with Mesh(("cta",), layout=(HEADS,), names=("head",)) as cta:
            padded_q = tf.insert_slice(
                tf.zeros(Tensor[(1, ceildiv(SEQ, BLOCK) * BLOCK, HEADS, HEAD_DIM), "bf16"]),
                q,
                (0, 0, 0, 0),
            )
            qp = tf.reshard(
                padded_q,
                (
                    1,
                    ceildiv(SEQ, BLOCK) * BLOCK,
                    HEADS @ cta.head,
                    HEAD_DIM,
                ),
                "smem",
            )
            output = tf.zeros(Tensor[(1, ceildiv(SEQ, BLOCK) * BLOCK, HEADS, HEAD_DIM), "bf16"])

            for query_start in range(0, ceildiv(SEQ, BLOCK) * BLOCK, BLOCK):
                qb = tf.slice(
                    qp,
                    (0, query_start, 0, 0),
                    sizes=(1, BLOCK, HEADS, HEAD_DIM),
                    strides=(1, 1, 1, 1),
                )
                queries = tf.transpose(tf.cast(qb, dtype="f32"), perm=(0, 2, 1, 3))
                scaled = queries * tf.full_like(queries, value=SCALE)
                template = tf.reduce(queries, axes=(-1,), keepdim=True, kind="sum")
                running_max = tf.full_like(template, value=-1e30)
                running_sum = tf.full_like(template, value=0.0)
                running_out = tf.full_like(queries, value=0.0)

                for key_start in range(0, ceildiv(SEQ, BLOCK) * BLOCK, BLOCK):
                    kb = tf.slice(
                        qp,
                        (0, key_start, 0, 0),
                        sizes=(1, BLOCK, HEADS, HEAD_DIM),
                        strides=(1, 1, 1, 1),
                    )
                    values = tf.transpose(tf.cast(kb, dtype="f32"), perm=(0, 2, 1, 3))
                    keys = tf.transpose(values, perm=(0, 1, 3, 2))
                    query_positions = (
                        tf.reshape(
                            tf.arange(Tensor[(BLOCK,), "i64", "gmem"]),
                            new_shape=(BLOCK, 1),
                        )
                        + query_start
                    )
                    key_positions = (
                        tf.reshape(
                            tf.arange(Tensor[(BLOCK,), "i64", "gmem"]),
                            new_shape=(1, BLOCK),
                        )
                        + key_start
                    )
                    keep = key_positions <= query_positions
                    raw_scores = tf.matmul(scaled, keys)
                    scores = tf.where(
                        keep,
                        raw_scores,
                        tf.full_like(raw_scores, value=-1e30),
                    )
                    block_max = tf.reduce(scores, axes=(-1,), keepdim=True, kind="max")
                    next_max = tf.max(running_max, block_max)
                    correction = tf.exp(running_max - next_max)
                    probabilities = tf.exp(scores - next_max)
                    next_sum = running_sum * correction + tf.reduce(
                        probabilities,
                        axes=(-1,),
                        keepdim=True,
                        kind="sum",
                    )
                    next_out = running_out * correction + tf.matmul(probabilities, values)
                    running_max = next_max
                    running_sum = next_sum
                    running_out = next_out

                attended = running_out / running_sum
                block_output = tf.transpose(tf.cast(attended, dtype="bf16"), perm=(0, 2, 1, 3))
                output = tf.insert_slice(output, block_output, (0, query_start, 0, 0))

            return tf.slice(
                output,
                (0, 0, 0, 0),
                sizes=(1, SEQ, HEADS, HEAD_DIM),
                strides=(1, 1, 1, 1),
            )


attend = PrefillDecodeAttention.entry_function()

__all__ = [
    "PrefillDecodeAttention",
    "attend",
    "SEQ",
    "CTX",
    "HEADS",
    "HEAD_DIM",
    "BLOCK",
]
