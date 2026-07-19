"""Fixed-shape Qwen online-softmax attention scan for CTA preflight."""

from __future__ import annotations

import math

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CudaTarget

S = 1
C = 4096
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
SCALE = 1.0 / math.sqrt(HEAD_DIM)


@func(target=CudaTarget(), topologies=(Topology("cta", 132),))
def qwen_static_online(
    q: Tensor[(1, S, NUM_Q_HEADS, HEAD_DIM), "bf16"],
    k_cache: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), "bf16"],
    v_cache: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), "bf16"],
) -> Tensor[(1, S, NUM_Q_HEADS, HEAD_DIM), "bf16"]:
    with Mesh(topology="cta", layout=Layout((132,), (1,))) as cta:  # noqa: F841
        q_sh = reshard(q, layout=(1, S, NUM_Q_HEADS, HEAD_DIM))
        q_f = tf.cast(q_sh, dtype="f32")
        q_s = q_f * tf.full_like(q_f, value=SCALE)
        tmpl = tf.reduce(
            q_f,
            axes=(-1,),
            keepdim=True,
            kind="sum",
        )
        m = tf.full_like(tmpl, value=-1e30)
        l = tf.full_like(tmpl, value=0.0)
        o = tf.full_like(q_f, value=0.0)
        for i in tile(4096):
            k_i = tf.reshape(
                tf.cast(
                    tf.repeat_interleave(
                        tf.gather(k_cache, i, axis=1),
                        repeats=GQA_GROUP,
                        axis=1,
                    ),
                    dtype="f32",
                ),
                new_shape=(1, 1, NUM_Q_HEADS, HEAD_DIM),
            )
            v_i = tf.reshape(
                tf.cast(
                    tf.repeat_interleave(
                        tf.gather(v_cache, i, axis=1),
                        repeats=GQA_GROUP,
                        axis=1,
                    ),
                    dtype="f32",
                ),
                new_shape=(1, 1, NUM_Q_HEADS, HEAD_DIM),
            )
            score = tf.reduce(
                q_s * k_i,
                axes=(-1,),
                keepdim=True,
                kind="sum",
            )
            m_new = tf.max(m, score)
            p = tf.exp(score - m_new)
            corr = tf.exp(m - m_new)
            l = l * corr + p
            o = o * corr + p * v_i
            m = m_new
        return tf.cast(o / l, dtype="bf16")


__all__ = [
    "C",
    "HEAD_DIM",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "S",
    "qwen_static_online",
]
