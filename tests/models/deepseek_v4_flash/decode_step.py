"""DeepSeek V4 flash decode step: HIR structural fixture only (no execution
or orchestration lives here).

``embed`` -> one decoder layer (attention + hash-router MoE) -> final
RMSNorm -> ``lm_head``. Real transformer **layer 0** structure throughout
(config.json ``compress_ratios[0] == 0``, pure sliding-window MLA); MoE is
``moe.py``'s hash-router variant (``moe_hash_gather`` /
``deepseek_v4_flash_moe_hash`` -- hash router = real layers 0..2 per
config.json's ``num_hash_layers``). The learned-router funcs (``moe_topk``,
``deepseek_v4_flash_moe``) remain defined in ``moe.py`` but are not part of
this module's tree.

Attention is two ``@func``s (``attn.mla_kv_update_v2`` then
``attn.mla_attend_v2``) rather than one composed ``@func`` -- see
``attention.py`` for why.

The module tree is a flat, single-level namespace: ``decode_step_module`` ->
{``attention``, ``moe``} -- no grandchildren (``shared_expert`` is folded
directly into ``moe_hash_module``'s own functions, not a nested child).
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash import attention as attn
from tests.models.deepseek_v4_flash.moe import (
    DIM,
    VOCAB,
    combine_expert_outputs,
    deepseek_v4_flash_moe_hash,
    moe_experts_core,
    moe_hash_gather,
    pre_moe_rms_norm,
    shared_expert,
    shared_fp8_dequant_w1,
    shared_fp8_dequant_w2,
)
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


attention_module = Module(
    name="attention",
    functions=(attn.mla_kv_update_v2, attn.mla_attend_v2),
    entry="mla_attend_v2",
)

moe_hash_module = Module(
    name="moe",
    functions=(pre_moe_rms_norm, moe_experts_core, moe_hash_gather,
               combine_expert_outputs, deepseek_v4_flash_moe_hash,
               shared_fp8_dequant_w1, shared_fp8_dequant_w2, shared_expert),
    entry="deepseek_v4_flash_moe_hash",
)

decode_step_module = Module(
    name="DeepSeekV4FlashDecodeStep",
    functions=(embed, residual_add, final_rms_norm, lm_head),
    modules=(attention_module, moe_hash_module),
    entry="lm_head",
)


__all__ = [
    "VOCAB",
    "attention_module",
    "decode_step_module",
    "embed",
    "final_rms_norm",
    "lm_head",
    "moe_hash_module",
    "residual_add",
]
