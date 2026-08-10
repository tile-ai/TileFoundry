"""Online-softmax GQA decode with context-length specialization.

Batch 1 decodes one token while query heads share KV heads. The ``pass``
prototype follows [hir §1.1](docs/spec/hir.md#11-function) and dispatches on a
half-open prior-cache length. Small contexts split query heads across CTAs;
large contexts compute split-KV partials and combine them, admitting the current
token exactly once. Cross-CTA handoff remains lowering work. Nondivisible block
sizes fail closed instead of dropping a tail.
"""

from __future__ import annotations

import math

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by the @func body
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op names for the @func body
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS


MAX_CTX = 262144


SMALL_CONTEXT_T = 4096


NUM_CTA = 8
NUM_SPLITS = NUM_CTA


S = 1
C = DimVar("ctx_len", 0, MAX_CTX)


CBLK = C // NUM_SPLITS

_D = HEAD_DIM
_HQ = NUM_Q_HEADS
_HKV = NUM_KV_HEADS
_G = GQA_GROUP
_SCALE = 1.0 / math.sqrt(HEAD_DIM)


@module(entry="gqa_online_attend", topologies=(Topology("cta", NUM_CTA),))
class GqaOnline:
    """The two decode strategies and the shared context-split kernels, in one execution domain.

    The two decode strategies and the shared context-split kernels, in one
    execution domain: an HIR Function may only call a Function its own Module
    owns, and every body here names the same ``cta`` level.
    """

    @func
    def gqa_online_attend(
        q: Tensor[(1, S, _HQ, _D), "bf16"],
        k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        k_new: Tensor[(1, S, _HKV, _D), "bf16"],
        v_new: Tensor[(1, S, _HKV, _D), "bf16"],
    ) -> Tensor[(1, S, _HQ, _D), "bf16"]:

        pass

    @gqa_online_attend.specialize(DimVarRangePat("ctx_len", 0, SMALL_CONTEXT_T))
    def head_on_cta(
        q: Tensor[(1, S, _HQ, _D), "bf16"],
        k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        k_new: Tensor[(1, S, _HKV, _D), "bf16"],
        v_new: Tensor[(1, S, _HKV, _D), "bf16"],
    ) -> Tensor[(1, S, _HQ, _D), "bf16"]:

        with Mesh(("cta",), layout=Layout((NUM_CTA,), (1,))) as cta:
            q_sh = reshard(q, layout=(1, S, _HQ @ cta, _D))
            q_f = tf.cast(q_sh, dtype="f32")
            q_s = q_f * tf.full_like(q_f, value=_SCALE)
            tmpl = tf.reduce(q_f, axes=(-1,), keepdim=True, kind="sum")
            m = tf.full_like(tmpl, value=-1e30)
            l = tf.full_like(tmpl, value=0.0)
            o = tf.full_like(q_f, value=0.0)
            for i in tile(C):
                k_i = tf.reshape(
                    tf.cast(
                        tf.repeat_interleave(tf.gather(k_cache, i, axis=1), repeats=_G, axis=1),
                        dtype="f32",
                    ),
                    new_shape=(1, 1, _HQ, _D),
                )
                v_i = tf.reshape(
                    tf.cast(
                        tf.repeat_interleave(tf.gather(v_cache, i, axis=1), repeats=_G, axis=1),
                        dtype="f32",
                    ),
                    new_shape=(1, 1, _HQ, _D),
                )
                score = tf.reduce(q_s * k_i, axes=(-1,), keepdim=True, kind="sum")
                m_new = tf.max(m, score)
                p = tf.exp(score - m_new)
                corr = tf.exp(m - m_new)
                l = l * corr + p
                o = o * corr + p * v_i
                m = m_new

            k_n = tf.cast(tf.repeat_interleave(k_new, repeats=_G, axis=2), dtype="f32")
            v_n = tf.cast(tf.repeat_interleave(v_new, repeats=_G, axis=2), dtype="f32")
            score_n = tf.reduce(q_s * k_n, axes=(-1,), keepdim=True, kind="sum")
            m_all = tf.max(m, score_n)
            p_n = tf.exp(score_n - m_all)
            corr_n = tf.exp(m - m_all)
            l = l * corr_n + p_n
            o = o * corr_n + p_n * v_n
            return tf.cast(o / l, dtype="bf16")

    @func
    def _ctx_partials(
        q: Tensor[(1, S, _HQ, _D), "bf16"],
        k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
    ):

        with Mesh(("cta",), layout=Layout((NUM_CTA,), (1,))) as cta:  # noqa: F841
            k_f = tf.transpose(
                tf.cast(
                    tf.repeat_interleave(
                        tf.reshape(k_cache, new_shape=(1, NUM_SPLITS, CBLK, _HKV, _D)),
                        repeats=_G,
                        axis=3,
                    ),
                    dtype="f32",
                ),
                perm=(0, 3, 1, 2, 4),
            )
            v_f = tf.transpose(
                tf.cast(
                    tf.repeat_interleave(
                        tf.reshape(v_cache, new_shape=(1, NUM_SPLITS, CBLK, _HKV, _D)),
                        repeats=_G,
                        axis=3,
                    ),
                    dtype="f32",
                ),
                perm=(0, 3, 1, 2, 4),
            )
            q_f = tf.cast(q, dtype="f32")
            q_s = q_f * tf.full_like(q_f, value=_SCALE)

            q_e = tf.reshape(q_s, new_shape=(1, S, _HQ, 1, 1, _D))
            k_e = tf.reshape(k_f, new_shape=(1, 1, _HQ, NUM_SPLITS, CBLK, _D))
            v_e = tf.reshape(v_f, new_shape=(1, 1, _HQ, NUM_SPLITS, CBLK, _D))
            scores = tf.reduce(q_e * k_e, axes=(-1,), keepdim=True, kind="sum")
            m_p = tf.reduce(scores, axes=(-2,), keepdim=True, kind="max")
            p = tf.exp(scores - m_p)
            l_p = tf.reduce(p, axes=(-2,), keepdim=True, kind="sum")
            o_p = tf.reduce(p * v_e, axes=(-2,), keepdim=False, kind="sum")
            return (
                tf.reshape(m_p, new_shape=(1, S, _HQ, NUM_SPLITS, 1)),
                tf.reshape(l_p, new_shape=(1, S, _HQ, NUM_SPLITS, 1)),
                o_p,
            )

    @func
    def _ctx_combine(
        m_p: Tensor[(1, S, _HQ, NUM_SPLITS, 1), "f32"],
        l_p: Tensor[(1, S, _HQ, NUM_SPLITS, 1), "f32"],
        o_p: Tensor[(1, S, _HQ, NUM_SPLITS, _D), "f32"],
        q: Tensor[(1, S, _HQ, _D), "bf16"],
        k_new: Tensor[(1, S, _HKV, _D), "bf16"],
        v_new: Tensor[(1, S, _HKV, _D), "bf16"],
    ) -> Tensor[(1, S, _HQ, _D), "bf16"]:

        with Mesh(("cta",), layout=Layout((NUM_CTA,), (1,))) as cta:  # noqa: F841
            m = tf.reduce(m_p, axes=(-2,), keepdim=True, kind="max")
            alpha = tf.exp(m_p - m)
            l = tf.reduce(alpha * l_p, axes=(-2,), keepdim=True, kind="sum")
            o = tf.reduce(alpha * o_p, axes=(-2,), keepdim=False, kind="sum")

            q_f = tf.cast(q, dtype="f32")
            q_s = q_f * tf.full_like(q_f, value=_SCALE)
            k_n = tf.cast(tf.repeat_interleave(k_new, repeats=_G, axis=2), dtype="f32")
            v_n = tf.cast(tf.repeat_interleave(v_new, repeats=_G, axis=2), dtype="f32")
            score_n = tf.reduce(q_s * k_n, axes=(-1,), keepdim=True, kind="sum")
            m_blk = tf.reshape(m, new_shape=(1, S, _HQ, 1))
            l_blk = tf.reshape(l, new_shape=(1, S, _HQ, 1))
            m_all = tf.max(m_blk, score_n)
            corr = tf.exp(m_blk - m_all)
            corr_n = tf.exp(score_n - m_all)
            return tf.cast((o * corr + corr_n * v_n) / (l_blk * corr + corr_n), dtype="bf16")

    @gqa_online_attend.specialize(DimVarRangePat("ctx_len", SMALL_CONTEXT_T, MAX_CTX))
    def ctx_split_kv(
        q: Tensor[(1, S, _HQ, _D), "bf16"],
        k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
        k_new: Tensor[(1, S, _HKV, _D), "bf16"],
        v_new: Tensor[(1, S, _HKV, _D), "bf16"],
    ) -> Tensor[(1, S, _HQ, _D), "bf16"]:

        m_p, l_p, o_p = _ctx_partials(q, k_cache, v_cache)
        return _ctx_combine(m_p, l_p, o_p, q, k_new, v_new)


gqa_online_attend = GqaOnline.lookup("gqa_online_attend")

__all__ = [
    "GqaOnline",
    "gqa_online_attend",
    "SMALL_CONTEXT_T",
    "NUM_SPLITS",
    "MAX_CTX",
    "HEAD_DIM",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "GQA_GROUP",
]
