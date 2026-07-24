"""DeepSeek V4 flash decode step: HIR structural fixture only (no execution
or orchestration lives here).

``embed`` -> one decoder layer (attention + hash-router MoE) -> final
RMSNorm -> ``lm_head``. Real transformer **layer 0** structure throughout
(config.json ``compress_ratios[0] == 0``, pure sliding-window MLA); MoE is
``moe.py``'s hash-router variant (hash router = real layers 0..2 per
config.json's ``num_hash_layers``). The learned-router funcs (``moe_topk``,
``deepseek_v4_flash_moe``) remain defined in ``moe.py`` but are not part of
this module's tree.

HuggingFace-modeling-file style: one component per file (``attention.py``,
``moe.py``), each assembling its own component ``Module``; this file holds
only the model head/tail funcs and the top-level ``Module`` that includes
them. The tree is flat and single-level: ``decode_step_module`` ->
{``attention``, ``moe``} -- no grandchildren (``shared_expert`` is folded
directly into ``moe_hash_module``'s own functions, not a nested child).
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.attention import attention_module
from tests.models.deepseek_v4_flash.moe import DIM, VOCAB, moe_hash_module
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.module import Module


@func
def embed(
    table: ConstTensor[(VOCAB, DIM), "bf16"],
    token_ids: Tensor[(1,), "i64"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.reshape(tf.gather(table, token_ids, axis=0), new_shape=(1, 1, DIM))


@func
def residual_add(
    a: Tensor[(1, 1, DIM), "bf16"],
    b: Tensor[(1, 1, DIM), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.add(a, b)


@func
def final_rms_norm(
    hidden: Tensor[(1, 1, DIM), "bf16"],
    weight: ConstTensor[(DIM,), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.rms_norm(hidden, weight)


@func
def lm_head(
    hidden: Tensor[(1, 1, DIM), "bf16"],
    weight: ConstTensor[(DIM, VOCAB), "bf16"],
) -> Tensor[(1, 1, VOCAB), "bf16"]:
    logits = tf.matmul(tf.reshape(hidden, new_shape=(1, DIM)), weight)
    return tf.reshape(logits, new_shape=(1, 1, VOCAB))


decode_step_module = Module(
    name="DeepSeekV4FlashDecodeStep",
    functions=(embed, residual_add, final_rms_norm, lm_head),
    modules=(attention_module, moe_hash_module),
    entry="lm_head",
)


__all__ = [
    "VOCAB",
    "decode_step_module",
    "embed",
    "final_rms_norm",
    "lm_head",
    "residual_add",
]
