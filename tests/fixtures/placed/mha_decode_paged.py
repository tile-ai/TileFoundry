"""Paged multi-head-attention decode, at each of its four manifest workloads.

From `tileops-loop-state-v2/round-2-mha-decode-paged`, whose `authored_hir.py`
hashes to sha256 e4f1c9ddb591adefebf3bf76c78ba627f4f17ad4ab4c256d4e297cc2a4bc14d5
and whose `findings.json` hashes to
sha256 41bdc13b13927159c9a83a4d82205f9415aeb75b9d835f29a2fb5490bcdb4c79. Nothing
here reads that round. There the four shapes came from an environment variable,
which a corpus cannot be asked: a fixture whose shape depends on the environment
is a different program on every machine. They are four Modules over one builder.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

_WAVE = 128


def _splits(batch: int, heads: int, kv: int) -> int:
    return max(1, min(_WAVE // (batch * heads), kv))


def _built(batch: int, heads: int, kv: int, dim: int, page: int, dtype: str):
    """One workload's Module, at the extents that workload states.

    A closure rather than four transcriptions: the manifest's four rows differ in
    five numbers and nothing else. One CTA per (sequence, head, KV split), each
    staging its KV rows in `smem` where TMA delivers them, reading them into
    `rmem` to do the arithmetic, and settling the split it reduced over on the
    return to `gmem`. Everything is authored inside the mesh because the paging
    arithmetic is part of the program; the finer slice a lane owns is not stated
    here at all, which is what `performance_findings.Levels` is about.
    """
    pages = -(-kv // page)
    split = _splits(batch, heads, kv)
    scale = dim**-0.5
    masked = -1.0e30

    @module(
        entry="mha_decode_paged",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", batch * heads * split),),
    )
    class MhaDecodePaged:
        @func
        def mha_decode_paged(
            q: Tensor[(batch, 1, heads, dim), dtype],
            k: Tensor[(kv, heads, dim), dtype],
            v: Tensor[(kv, heads, dim), dtype],
            real_seqlen_kv: Tensor[(batch,), "i32"],
            block_table: Tensor[(batch, pages), "i32"],
        ) -> Tensor[(batch, 1, heads, dim), dtype]:
            with Mesh(
                ("cta",),
                layout=(batch, heads, split),
                names=("seq", "head", "split"),
            ) as cta:
                k_cache = tf.reshard(k, (kv, heads, dim), "gmem")
                v_cache = tf.reshard(v, (kv, heads, dim), "gmem")
                table = tf.reshard(block_table, (batch, pages), "gmem")
                lengths = tf.reshard(real_seqlen_kv, (batch,), "gmem")
                q_placed = tf.reshard(q, (batch, 1, heads, dim), "gmem")

                gathered = tf.reshape(table, new_shape=(batch * pages,))
                k_logical = tf.reshape(
                    tf.index_select(
                        tf.reshape(k_cache, new_shape=(pages, page, heads, dim)),
                        gathered,
                        dim=0,
                    ),
                    new_shape=(batch, 1, kv, heads, dim),
                )
                v_logical = tf.reshape(
                    tf.index_select(
                        tf.reshape(v_cache, new_shape=(pages, page, heads, dim)),
                        gathered,
                        dim=0,
                    ),
                    new_shape=(batch, 1, kv, heads, dim),
                )
                q_rows = tf.reshape(q_placed, new_shape=(batch, 1, 1, heads, dim))

                k_tile = tf.reshard(
                    k_logical,
                    (batch @ cta.seq, 1, kv @ cta.split, heads @ cta.head, dim),
                    "smem",
                )
                v_tile = tf.reshard(
                    v_logical,
                    (batch @ cta.seq, 1, kv @ cta.split, heads @ cta.head, dim),
                    "smem",
                )

                q_reg = tf.reshard(
                    q_rows, (batch @ cta.seq, 1, 1, heads @ cta.head, dim), "rmem"
                )
                k_reg = tf.reshard(
                    k_tile,
                    (batch @ cta.seq, 1, kv @ cta.split, heads @ cta.head, dim),
                    "rmem",
                )
                v_reg = tf.reshard(
                    v_tile,
                    (batch @ cta.seq, 1, kv @ cta.split, heads @ cta.head, dim),
                    "rmem",
                )

                q32 = tf.cast(q_reg, dtype="f32")
                k32 = tf.cast(k_reg, dtype="f32")
                v32 = tf.cast(v_reg, dtype="f32")
                raw = tf.reduce(q32 * k32, axes=(-1,), keepdim=True, kind="sum")
                score = raw * tf.full_like(raw, value=scale)

                position = tf.reshard(
                    tf.reshape(
                        tf.arange(type=Tensor[(kv,), "i32"]),
                        new_shape=(1, 1, kv, 1, 1),
                    ),
                    (1, 1, kv @ cta.split, 1, 1),
                    "rmem",
                )
                length = tf.reshard(
                    tf.reshape(lengths, new_shape=(batch, 1, 1, 1, 1)),
                    (batch @ cta.seq, 1, 1, 1, 1),
                    "rmem",
                )
                live = tf.where(
                    tf.cmp_lt(position, length),
                    score,
                    tf.full_like(score, value=masked),
                )

                peak = tf.reduce(live, axes=(-3,), keepdim=True, kind="max")
                weight = tf.exp(live - peak)
                total = tf.reduce(weight, axes=(-3,), keepdim=False, kind="sum")
                blended = tf.reduce(weight * v32, axes=(-3,), keepdim=False, kind="sum")
                out = tf.cast(blended / total, dtype=dtype)

                return tf.reshard(
                    out, (batch @ cta.seq, 1, heads @ cta.head, dim), "gmem"
                )

    return MhaDecodePaged


SingleTokenPage128 = _built(1, 16, 512, 128, 128, "f16")
Batch2Page256 = _built(2, 8, 1024, 64, 256, "f16")
LongerCache = _built(1, 8, 1024, 64, 256, "f16")
ShorterCache = _built(1, 8, 512, 64, 256, "f16")

CACHE_PAIR = (ShorterCache, LongerCache)
