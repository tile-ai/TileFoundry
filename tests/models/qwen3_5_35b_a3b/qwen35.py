"""Qwen3.5-35B-A3B decode-step model, HuggingFace-modeling-file style: one
component per file (``attention.py`` / ``gdn.py`` / ``moe.py``), each
assembling its own component ``Module``; this file holds the ``embed`` /
``head`` boundary funcs and the top-level ``Module`` tree. The tree is flat
and single-level: ``qwen35_module`` -> {``attention``, ``gdn``, ``moe``} --
no grandchildren.

Mirrors ``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` (transformers
5.12.1; cited as M in this file and the component files). 40 layers = 30 x
linear_attention (GatedDeltaNet, ``gdn.py``) + 10 x full_attention
(``attention.py``), per config.json's ``layer_types``; every layer's MoE
(``moe.py``) is 256 experts top-8 + a scalar-gated shared expert.
hidden=2048, vocab=248320.

Each ``@func`` is a fusion boundary: one kernel's semantic contract (a
future ``RuntimeModule`` method). RMSNorm weights (M:817) are checkpoint-
stored as ``w - 1``; the three mix funcs (``full_attn_mix``/``gdn_mix``/
``moe_mix``) take the RAW (not +1) gamma and apply ``(1 + w)`` in-body,
while each component's ``*_convert`` func is the sole RAW-checkpoint ->
CANONICAL (kernel-native layout) conversion entry point -- pure repack/
cast, no numeric change.
"""
from __future__ import annotations

from tests.models.qwen3_5_35b_a3b.attention import attention_module
from tests.models.qwen3_5_35b_a3b.config import HIDDEN, VOCAB
from tests.models.qwen3_5_35b_a3b.gdn import gdn_module
from tests.models.qwen3_5_35b_a3b.moe import moe_module
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.module import Module


@func
def embed(
    token_id: Tensor[(1,), "i32"],
    embed_tokens: ConstTensor[(VOCAB, HIDDEN), "bf16"],
):
    # Residual-stream start: gather one embedding-table row by token id.
    h = tf.gather(embed_tokens, token_id, axis=0)         # (1, HIDDEN) bf16
    return tf.reshape(h, new_shape=(1, 1, HIDDEN))


@func
def head(
    x: Tensor[(1, 1, HIDDEN), "bf16"],
    final_norm_raw: ConstTensor[(HIDDEN,), "f32"],        # RAW; (1+w) applied in-body
    lm_head: ConstTensor[(VOCAB, HIDDEN), "bf16"],        # native [vocab,hidden]
) -> Tensor[(1, VOCAB), "f32"]:
    # final RMSNorm + lm_head GEMV -> f32 logits. matmul(lm_head[vocab,hidden],
    # h_col[hidden,1]) avoids transposing the (large) lm_head weight.
    h = tf.rms_norm(x, 1.0 + final_norm_raw)              # (1,1,HIDDEN) bf16
    h_col = tf.reshape(tf.cast(h, dtype="f32"), new_shape=(HIDDEN, 1))
    logits = tf.matmul(tf.cast(lm_head, dtype="f32"), h_col)   # (VOCAB, 1) f32
    return tf.reshape(logits, new_shape=(1, VOCAB))


qwen35_module = Module(
    name="Qwen35_35B_A3B",
    functions=(embed, head),
    modules=(attention_module, gdn_module, moe_module),
    entry="head",
)
