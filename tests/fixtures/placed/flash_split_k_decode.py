"""Split-K decode attention over a two-dimensional CTA mesh.

The head axis owns one attention head per CTA group and the worker axis splits
the context. Each worker starts at its own ``here * BLOCK`` offset and advances
by ``BLOCK * WORKERS``, so its K/V windows are disjoint and the workers cover
the complete context together. The loop keeps an online ``(m, l, acc)`` state
per worker. A second reshard broadcasts those slots across the worker axis;
the final max and weighted sums are explicit log-sum-exp combination rather
than a ``Partial`` value.
"""

from __future__ import annotations

import math

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

HEADS = 16
WORKERS = 8
HEAD_DIM = 64
BLOCK = 128
CTX = DimVar("ctx", 1, 65_537)
SCALE = 1.0 / math.sqrt(HEAD_DIM)


@module(
    entry="flash_split_k_decode",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", HEADS * WORKERS),),
)
class FlashSplitKDecode:
    @func
    def flash_split_k_decode(
        q: Tensor[(1, 1, HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, HEADS, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HEADS, HEAD_DIM), "bf16"]:
        with Mesh(
            ("cta",), layout=(HEADS, WORKERS), names=("head", "w")
        ) as cta:
            qh = tf.reshard(
                q, (1, 1, HEADS @ cta.head, HEAD_DIM), "smem"
            )
            queries = tf.transpose(tf.cast(qh, dtype="f32"), perm=(0, 2, 1, 3))
            scaled = queries * tf.full_like(queries, value=SCALE)

            m_slots = tf.reshard(
                tf.zeros(shape=(WORKERS, HEADS, 1, 1), dtype="f32"),
                (WORKERS @ cta.w, HEADS @ cta.head, 1, 1),
                "smem",
            )
            acc_slots = tf.reshard(
                tf.zeros(shape=(WORKERS, HEADS, 1, HEAD_DIM), dtype="f32"),
                (WORKERS @ cta.w, HEADS @ cta.head, 1, HEAD_DIM),
                "smem",
            )
            m = tf.full_like(m_slots, value=-1e30)
            l = tf.full_like(m_slots, value=0.0)
            acc = tf.full_like(acc_slots, value=0.0)

            for c in tile(CTX, BLOCK * WORKERS):
                base = c + cta.w * BLOCK
                kb = tf.reshard(
                    k_cache[:, base : base + BLOCK, :, :],
                    (1, BLOCK, HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                vb = tf.reshard(
                    v_cache[:, base : base + BLOCK, :, :],
                    (1, BLOCK, HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                keys = tf.transpose(
                    tf.cast(kb, dtype="f32"), perm=(0, 2, 3, 1)
                )
                values = tf.transpose(
                    tf.cast(vb, dtype="f32"), perm=(0, 2, 1, 3)
                )
                scores = tf.matmul(scaled, keys)
                block_m = tf.reduce(
                    scores, axes=(-1,), keepdim=True, kind="max"
                )
                next_m = tf.max(m, block_m)
                correction = tf.exp(m - next_m)
                weights = tf.exp(scores - next_m)
                l = l * correction + tf.reduce(
                    weights, axes=(-1,), keepdim=True, kind="sum"
                )
                acc = acc * correction + tf.matmul(weights, values)
                m = next_m

            all_m = tf.reshard(
                m, (WORKERS, HEADS @ cta.head, 1, 1), "smem"
            )
            all_l = tf.reshard(
                l, (WORKERS, HEADS @ cta.head, 1, 1), "smem"
            )
            all_acc = tf.reshard(
                acc, (WORKERS, HEADS @ cta.head, 1, HEAD_DIM), "smem"
            )
            global_m = tf.reduce(all_m, axes=(0,), keepdim=False, kind="max")
            weights = tf.exp(all_m - global_m)
            global_l = tf.reduce(
                weights * all_l, axes=(0,), keepdim=False, kind="sum"
            )
            global_acc = tf.reduce(
                weights * all_acc, axes=(0,), keepdim=False, kind="sum"
            )
            output = tf.cast(global_acc / global_l, dtype="bf16")
            output = tf.reshape(output, new_shape=(1, 1, HEADS, HEAD_DIM))
            return tf.reshard(
                output, (1, 1, HEADS @ cta.head, HEAD_DIM), "gmem"
            )


flash_split_k_decode = FlashSplitKDecode.entry_function()

__all__ = [
    "FlashSplitKDecode",
    "flash_split_k_decode",
    "BLOCK",
    "CTX",
    "HEADS",
    "WORKERS",
]
