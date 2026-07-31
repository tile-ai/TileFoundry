"""Kimi-Linear-48B-A3B's three distinct submodules as tilefoundry IR Modules.

Three Modules rather than one decoder, because this model's minimum content is a
set of *kinds*, not a stack: one KDA layer, one MLA layer, and the MoE they share.
The public model repeats KDA three times for every MLA; repeating that here would
add no semantics, so ``config.MINIMUM_LAYERS`` names the two 0-based indices that
do -- layer 0 (KDA + dense MLP) and layer 3 (MLA + MoE).

Component-only on purpose: no decoder layer, no layer stack, no token-ids-to-logits
root. The root composes the three kinds so they can be selected, not run as a step;
``reference.KDA_BLOCK_REASON`` records that one of those kinds has no runnable
reference, so a stack here could not be scored against anything.

Decode, one token per step. ``S`` is the literal 1 and ``ctx_len`` is the only
range, exactly as the rest of the corpus states it.

The KV cache is explicit tensors in and out, and for the two attention kinds it
means two different things. That is a per-submodule detail, not two contracts:

- ``mla_attention`` reads ``ctx_len`` prior positions and returns this token's
  own key and value. The caller **appends**. Every config is expressed in
  ``ctx_len`` alone, so nothing here has an axis of ``ctx_len + 1``.
- ``kda_attention`` reads a fixed-size recurrent state and a fixed-size
  convolution window and returns both updated. The caller **replaces**. Nothing
  in it depends on ``ctx_len`` at all.

A recurrent state carrying no ``ctx_len`` is strictly more constrained than the
decode contract, not in conflict with it: the invariant the contract states is
that the step reads the past as explicit tensors and hands back explicit tensors,
with no Hugging Face ``Cache`` on either side, and that holds in both cases.
Append versus replace is the submodule's business. The next model with a
recurrent state does not need to re-litigate this.

What a KDA layer's "cache" *is*, concretely: a `[heads, v_dim, k_dim]` recurrent
state matrix (the delta-rule memory, 32x128x128 here), plus one
`short_conv_kernel_size - 1` = 3-deep window of the previous tokens' q/k/v
*projections*, because q, k and v each pass a causal depthwise convolution of
kernel 4 before the recurrence. The window is a genuine per-position cache, but
of a fixed depth -- append-and-evict, not growth. Neither piece can be gathered
from a longer context: the state is a summary of every prior token, so producing
it for a ``ctx_len``-long context means running the recurrence over that context.

Kimi's MLA is NoPE (``mla_use_nope: true``), which does not remove the 64
rotary-half dimensions -- it stops rotating them. They still enter the score and
the ``qk_head_dim = 192`` scaling denominator. So ``mla_attention`` is authored
once, with the rotary always applied, and NoPE is driven by handing it
``cos = 1, sin = 0``; that this is exactly the identity is measured in
``test_mla.py``, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

# ── the checkpoint's own configuration class ─────────────────────────────────
#
# Vendored, byte for byte, from `configuration_kimi.py` in
# moonshotai/Kimi-Linear-48B-A3B-Instruct at revision
# e1df551a447157d4658b573f9a695d57658590e9 -- the file the published config's
# `auto_map` points at. `transformers` 5.14.1 has no `kimi_linear`: the model type
# is absent from `CONFIG_MAPPING`, and `AutoConfig` refuses the directory without
# `trust_remote_code` and cannot find the module with it. So the class that reads
# this checkpoint is this one, and the choice is to copy it or to counterfeit it.
#
# It is the *config* and nothing else: it imports only `PretrainedConfig`, defines
# no layers, and executes nothing at import. The modelling code stays unvendored
# and unexecuted; every oracle in `reference.py` is a different published model's
# Hugging Face module shown to compute the same function.


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )


# ── this checkpoint ──────────────────────────────────────────────────────────


def published(path: Path | None = None) -> KimiLinearConfig:
    """The checkpoint's own configuration, read by the class it names.

    The file sits beside this module, so a copy of this directory carries its own
    dimensions and needs nothing importable around it.
    """
    path = Path(__file__).parent / "config.json" if path is None else path
    return KimiLinearConfig(**json.loads(path.read_text(encoding="utf-8")))


#: Envelope for the one dynamic dimension, and the position table's extent. Ours,
#: not published: `max_position_embeddings` is 1048576, and the envelope only has
#: to contain the lengths anything here is asked about -- a million-long bound
#: costs analysis precision for nothing.
MAX_CTX = MAX_POS = 4096


def build_kimi_linear_48b_a3b(config: KimiLinearConfig):
    """This model at *config*: its three kernels, as the children of one root.

    Public because the model is asked about at more than one expert count, and a
    ``@module`` class body at file scope is evaluated once. Callers name the config
    they mean -- ``published()``, or the reduced-expert one ``reference.py`` builds
    for the MoE oracle -- and get a tree that shares no IR node with any other
    call's. Nothing is read or evaluated at import.
    """
    # The prior cache this step reads: the only range this model carries, and only
    # the MLA submodule carries it. Zero is a first step, and the exclusive upper
    # bound is MAX_CTX, which is also the position table's extent.
    C = DimVar("ctx_len", 0, MAX_CTX)

    # One token per step.
    S = 1

    _H = config.num_attention_heads
    _NOPE = config.qk_nope_head_dim       # 128
    _ROPE = config.qk_rope_head_dim       # 64
    _QK = _NOPE + _ROPE                   # 192, the score dim and so the scaling one
    _V = config.v_head_dim                # 128
    _KVB = _NOPE + _V                     # 256, kv_b_proj's per-head output

    # KDA's dimensions are published nested, and its head_dim is not the top-level
    # one: `head_dim: 72` is hidden_size // num_attention_heads and is read by
    # neither path -- KDA uses 128 and MLA uses 192 (q/k) and 128 (v).
    _KDA = config.linear_attn_config
    _KH = _KDA["num_heads"]               # 32
    _KD = _KDA["head_dim"]                # 128
    _KP = _KH * _KD                       # 4096
    _W = _KDA["short_conv_kernel_size"]   # 4
    _WS = _W - 1                          # 3 stored positions

    _E = config.num_experts
    _TOPK = config.num_experts_per_token
    _MI = config.moe_intermediate_size
    _SI = _MI * config.num_shared_experts

    _EPS = config.rms_norm_eps

    # The published dtype as the DSL spells it. The checkpoint stores its weights
    # at this precision, so it is what a kernel reading them consumes.
    _DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
        str(config.dtype).removeprefix("torch.")
    ]


    @module(entry="mla_attention")
    class KimiMla:
        """Layer 3's attention: multi-head latent attention, decode step.

        Mirrors ``DeepseekV3Attention.forward`` at Kimi's ranks, which is the same
        parameter set and the same score scaling vLLM's ``KimiMLAAttention`` builds.
        """

        @func
        def mla_attention(
            hidden: Tensor[(1, S, config.hidden_size), _DT],
            gamma_in: Tensor[(config.hidden_size,), _DT],
            w_q: Tensor[(1, config.hidden_size, (_H * _QK)), _DT],
            w_kv_a: Tensor[(1, config.hidden_size, (config.kv_lora_rank + _ROPE)), _DT],
            gamma_kv_a: Tensor[(config.kv_lora_rank,), _DT],
            w_kv_b: Tensor[(1, config.kv_lora_rank, (_H * _KVB)), _DT],
            cos_cache: Tensor[(MAX_POS, config.qk_rope_head_dim), _DT],
            sin_cache: Tensor[(MAX_POS, config.qk_rope_head_dim), _DT],
            pos_ids: Tensor[(S,), "i32"],
            k_cache: Tensor[(1, C, config.num_attention_heads, _QK), _DT],
            v_cache: Tensor[(1, C, config.num_attention_heads, config.v_head_dim), _DT],
            scale: Tensor[(1, 1, 1, 1), _DT],
            w_o: Tensor[(1, (_H * _V), config.hidden_size), _DT],
        ):
            # Fused input RMSNorm + MLA, no residual (the layer owns that). Returns
            # the attention output with this token's key and value for the caller to
            # append.
            # `DeepseekV3RMSNorm`, which this model's MLA is measured against, ends
            # `self.weight * hidden_states.to(input_dtype)`: the normalised value is
            # rounded to the input dtype before the learned scale multiplies it.
            # `tf.rms_norm` is the generic op and keeps f32 through that multiply.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_in

            # The query is a plain projection here: q_lora_rank is null, so there is
            # no q_a/q_b pair to fold.
            q = tf.reshape(tf.matmul(hn, w_q), new_shape=(1, S, _H, _QK))
            q_pass = q[:, :, :, :_NOPE]
            q_rot = q[:, :, :, _NOPE:_QK]

            # One projection yields the latent and the rope part together, and the
            # rope part is shared across heads -- that is the "MQA" in
            # kv_a_proj_with_mqa.
            compressed = tf.matmul(hn, w_kv_a)
            latent = compressed[:, :, : config.kv_lora_rank]
            k_rot_shared = compressed[:, :, config.kv_lora_rank : (config.kv_lora_rank + _ROPE)]

            kv_n32 = tf.cast(latent, dtype="f32")
            kv_n_var = tf.reduce(kv_n32 * kv_n32, axes=(-1,), keepdim=True, kind="mean")
            kv_n = tf.cast(kv_n32 * tf.rsqrt(kv_n_var + _EPS), dtype="bf16") * gamma_kv_a
            kv = tf.reshape(
                tf.matmul(kv_n, w_kv_b),
                new_shape=(1, S, _H, _KVB),
            )
            k_nope = kv[:, :, :, :_NOPE]
            v_new = kv[:, :, :, _NOPE:_KVB]

            # Rotate the shared 64-wide part once, then broadcast it over the heads;
            # repeat_interleave on a length-1 axis is that broadcast.
            k_rot_1 = tf.reshape(k_rot_shared, new_shape=(1, S, 1, _ROPE))
            _kq, k_rot = tf.rope(k_rot_1, k_rot_1, cos_cache, sin_cache, pos_ids)
            k_rot_h = tf.repeat_interleave(k_rot, repeats=_H, axis=2)
            q_rot_r, _kr = tf.rope(q_rot, q_rot, cos_cache, sin_cache, pos_ids)

            q_full = tf.concat(q_pass, q_rot_r, axis=-1)
            k_new = tf.concat(k_nope, k_rot_h, axis=-1)

            # Online softmax over two differently shaped score groups: the cache and
            # the token itself. No mask -- one query at the end of the context may
            # attend every position there is. Every key/value head serves exactly one
            # query head here (num_key_value_heads == num_attention_heads), so there
            # is no GQA expansion.
            q_s = q_full * scale
            k_ctx = tf.reshape(
                tf.transpose(k_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _QK)
            )
            v_ctx = tf.reshape(
                tf.transpose(v_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _V)
            )
            q_e = tf.reshape(q_s, new_shape=(1, S, _H, 1, _QK))
            score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
            score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

            peak = tf.max(
                tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
            )
            peak_e = tf.reshape(peak, new_shape=(1, S, _H, 1, 1))
            p_ctx = tf.exp(score_ctx - peak_e)
            p_new = tf.exp(score_new - peak)
            total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
            weighted = (
                tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
                + p_new * v_new
            )
            attn = weighted / total
            out = tf.matmul(tf.reshape(attn, new_shape=(1, S, (_H * _V))), w_o)
            return out, k_new, v_new


    @module(entry="kda_attention")
    class KimiKda:
        """Layer 0's attention: Kimi Delta Attention, decode step.

        A gated delta rule whose forget gate is **per channel**, which is what makes
        it KDA rather than the scalar-per-head gated delta net that ships as
        ``Qwen3NextGatedDeltaNet``: there, ``g`` is one number per head and the state
        decays uniformly; here ``g`` is a 128-wide vector per head and the state
        decays column by column. That difference is why Qwen3-Next is not a stand-in
        oracle for this submodule.

        Transcribed from vLLM 0.18.0's ``KimiDeltaAttention.forward`` and the
        ``fused_recurrent_kda`` kernel body, both read on disk. **The values this
        computes are unverified** -- see ``reference.py`` for why there is no oracle.
        """

        @func
        def short_conv(
            x: Tensor[(1, S, _KP), _DT],
            conv_w: Tensor[(_W, _KP), _DT],
            conv_state: Tensor[(1, _WS, _KP), _DT],
        ):
            # Causal depthwise convolution of kernel 4 with a silu, evaluated for one
            # token: the window is the three stored positions followed by this one, so
            # the convolution is a weighted sum over the window's time axis rather
            # than a sliding op. Returns the activation and the window to store next,
            # which is this window with its oldest position dropped.
            window = tf.concat(conv_state, x, axis=1)
            acc = tf.reduce(
                window
                * tf.reshape(
                    conv_w,
                    new_shape=(1, _W, _KP),
                ),
                axes=(1,),
                keepdim=True,
                kind="sum",
            )
            out = tf.silu(acc)
            state_next = window[:, 1 : _W, :]
            return out, state_next

        @func
        def l2_normalize(
            x: Tensor[(1, S, _KH, _KD), _DT],
        ) -> Tensor[(1, S, _KH, _KD), _DT]:
            # x / sqrt(sum(x*x) + 1e-6), per head. The epsilon sits inside the square
            # root, matching the kernel; it is not an rms_norm, which would divide by
            # the *mean* of the squares and carry a weight.
            sq = tf.reduce(tf.square(x), axes=(-1,), keepdim=True, kind="sum")
            return x * tf.rsqrt(sq + tf.full_like(sq, value=1e-6))

        @func
        def kda_gate(
            hidden_norm: Tensor[(1, S, config.hidden_size), _DT],
            w_f_a: Tensor[(1, config.hidden_size, _KD), _DT],
            w_f_b: Tensor[(1, _KD, _KP), _DT],
            dt_bias: Tensor[(_KP,), _DT],
            a_log: Tensor[(_KH,), _DT],
        ) -> Tensor[(1, S, _KH, _KD), _DT]:
            # The per-channel forget gate: a low-rank projection through
            # kda_head_dim, biased, softplus'd, and scaled by -exp(A_log) per head.
            # softplus here is beta=1, which is what the kernel computes; the kernel's
            # threshold=20 switch to the linear branch is a numerical guard on the
            # same function, not a different one.
            low = tf.matmul(hidden_norm, w_f_a)
            g_raw = tf.reshape(
                tf.matmul(low, w_f_b) + dt_bias,
                new_shape=(1, S, _KH, _KD),
            )
            decay_rate = -tf.exp(tf.reshape(a_log, new_shape=(1, 1, _KH, 1)))
            return decay_rate * tf.softplus(g_raw)

        @func
        def kda_attention(
            hidden: Tensor[(1, S, config.hidden_size), _DT],
            gamma_in: Tensor[(config.hidden_size,), _DT],
            w_q: Tensor[(1, config.hidden_size, _KP), _DT],
            w_k: Tensor[(1, config.hidden_size, _KP), _DT],
            w_v: Tensor[(1, config.hidden_size, _KP), _DT],
            conv_w_q: Tensor[(_W, _KP), _DT],
            conv_w_k: Tensor[(_W, _KP), _DT],
            conv_w_v: Tensor[(_W, _KP), _DT],
            conv_state_q: Tensor[(1, _WS, _KP), _DT],
            conv_state_k: Tensor[(1, _WS, _KP), _DT],
            conv_state_v: Tensor[(1, _WS, _KP), _DT],
            w_f_a: Tensor[(1, config.hidden_size, _KD), _DT],
            w_f_b: Tensor[(1, _KD, _KP), _DT],
            dt_bias: Tensor[(_KP,), _DT],
            a_log: Tensor[(_KH,), _DT],
            w_b: Tensor[(1, config.hidden_size, _KH), _DT],
            w_g_a: Tensor[(1, config.hidden_size, _KD), _DT],
            w_g_b: Tensor[(1, _KD, _KP), _DT],
            gamma_o: Tensor[(_KD,), _DT],
            w_o: Tensor[(1, _KP, config.hidden_size), _DT],
            state: Tensor[
                (1, _KH, _KD, _KD), _DT
            ],
            scale: Tensor[(1, 1, 1), _DT],
        ):
            # One decode step. Returns the output, the updated recurrent state, and
            # the three updated convolution windows -- all fixed size, none carrying
            # ctx_len. The caller replaces rather than appends.
            # `DeepseekV3RMSNorm`, which this model's MLA is measured against, ends
            # `self.weight * hidden_states.to(input_dtype)`: the normalised value is
            # rounded to the input dtype before the learned scale multiplies it.
            # `tf.rms_norm` is the generic op and keeps f32 through that multiply.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_in

            q_c, conv_q_next = short_conv(tf.matmul(hn, w_q), conv_w_q, conv_state_q)
            k_c, conv_k_next = short_conv(tf.matmul(hn, w_k), conv_w_k, conv_state_k)
            v_c, conv_v_next = short_conv(tf.matmul(hn, w_v), conv_w_v, conv_state_v)

            q_h = tf.reshape(q_c, new_shape=(1, S, _KH, _KD))
            k_h = tf.reshape(k_c, new_shape=(1, S, _KH, _KD))
            v_h = tf.reshape(v_c, new_shape=(1, S, _KH, _KD))

            # l2 normalisation happens inside the kernel, before the scale.
            q_n = l2_normalize(q_h)
            k_n = l2_normalize(k_h)
            q_s = tf.reshape(q_n, new_shape=(1, _KH, 1, _KD)) * scale

            g = kda_gate(hn, w_f_a, w_f_b, dt_bias, a_log)
            beta = tf.reshape(
                tf.sigmoid(tf.matmul(hn, w_b)), new_shape=(1, _KH, 1)
            )

            # The delta rule, one token. The state is [heads, v_dim, k_dim]; the decay
            # multiplies it column-wise along k_dim, which is the per-channel gate.
            decay = tf.reshape(tf.exp(g), new_shape=(1, _KH, 1, _KD))
            k_r = tf.reshape(k_n, new_shape=(1, _KH, 1, _KD))
            h_decayed = state * decay
            kv_mem = tf.reduce(h_decayed * k_r, axes=(-1,), keepdim=False, kind="sum")
            delta = (tf.reshape(v_h, new_shape=(1, _KH, _KD)) - kv_mem) * beta
            state_next = (
                h_decayed + tf.reshape(delta, new_shape=(1, _KH, _KD, 1)) * k_r
            )
            attn = tf.reduce(state_next * q_s, axes=(-1,), keepdim=False, kind="sum")

            # Gated output norm: rms_norm(attn) * sigmoid(g2), the "sigmoid" activation
            # of the kernel's fused gated RMSNorm -- not a swish, which would be
            # g2 * sigmoid(g2).
            g2 = tf.reshape(
                tf.matmul(tf.matmul(hn, w_g_a), w_g_b), new_shape=(1, _KH, _KD)
            )
            gated = tf.rms_norm(attn, gamma_o, eps=_EPS) * tf.sigmoid(g2)
            out = tf.matmul(tf.reshape(gated, new_shape=(1, S, _KP)), w_o)
            return out, state_next, conv_q_next, conv_k_next, conv_v_next


    @module(entry="moe")
    class KimiMoe:
        """The MoE both non-dense layers share: sigmoid router, 256 experts, top-8,
        one shared expert.

        Routing has one subtlety that no config check catches. Selection reads
        ``sigmoid(logits) + e_score_correction_bias``; the routing weights are
        gathered from the *unbiased* sigmoid scores. So the bias moves *which*
        experts run without appearing in *how much* they count. Recovering the
        unbiased score as ``biased_top_value - bias[index]`` is exact and needs only
        an axis-0 gather, which is why there is no second gather over the score row.

        ``num_expert_group = topk_group = 1`` makes the published grouped-top-k the
        identity -- one group holds every expert, so nothing is ever masked out.
        There is no group stage here because at these numbers it has nothing to do.
        """

        @func
        def router(
            tokens: Tensor[(S, config.hidden_size), _DT],
            w_router: Tensor[(config.hidden_size, _E), _DT],
            bias: Tensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
        ):
            # f32 throughout: selection has to agree with the oracle's, and a top-k
            # over bf16 scores can tie differently.
            logits = tf.cast(tf.matmul(tokens, w_router), dtype="f32")
            scores = tf.sigmoid(logits)
            biased = scores + tf.cast(bias, dtype="f32")
            top_biased, indices = tf.topk(biased, k=config.num_experts_per_token, axis=-1)
            # The weights come from the unbiased scores; subtracting the selected
            # experts' bias recovers them exactly.
            unbiased = top_biased - tf.cast(tf.gather(bias, indices, axis=0), dtype="f32")
            denom = tf.reduce(unbiased, axes=(-1,), keepdim=True, kind="sum")
            # normalise, *then* scale: moe_renormalize is true and the scaling factor
            # is applied to the normalised weights, not folded into the denominator.
            weights = unbiased / denom * tf.cast(routed_scale, dtype="f32")
            return tf.cast(weights, dtype=_DT), indices

        @func
        def shared_expert(
            tokens: Tensor[(S, config.hidden_size), _DT],
            w_gate: Tensor[(1, config.hidden_size, _SI), _DT],
            w_up: Tensor[(1, config.hidden_size, _SI), _DT],
            w_down: Tensor[(1, _SI, config.hidden_size), _DT],
        ) -> Tensor[(S, config.hidden_size), _DT]:
            # One dense SwiGLU expert every token pays for, unscaled: the routed
            # scaling factor applies to the routed branch only.
            x = tf.reshape(tokens, new_shape=(1, S, config.hidden_size))
            gate = tf.matmul(x, w_gate)
            up = tf.matmul(x, w_up)
            h = tf.silu(gate) * up
            return tf.reshape(tf.matmul(h, w_down), new_shape=(S, config.hidden_size))

        @func
        def moe(
            hidden: Tensor[(1, S, config.hidden_size), _DT],
            gamma_post: Tensor[(config.hidden_size,), _DT],
            w_router: Tensor[(config.hidden_size, _E), _DT],
            bias: Tensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
            w_gate: Tensor[(_E, config.moe_intermediate_size, config.hidden_size), _DT],
            w_up: Tensor[(_E, config.moe_intermediate_size, config.hidden_size), _DT],
            w_down: Tensor[(_E, config.hidden_size, config.moe_intermediate_size), _DT],
            sh_gate: Tensor[(1, config.hidden_size, _SI), _DT],
            sh_up: Tensor[(1, config.hidden_size, _SI), _DT],
            sh_down: Tensor[(1, _SI, config.hidden_size), _DT],
        ) -> Tensor[(1, S, config.hidden_size), _DT]:
            # Fused post-attention RMSNorm + MoE, no residual (the layer owns that).
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_post
            tokens = tf.reshape(hn, new_shape=(S, config.hidden_size))
            weights, indices = router(tokens, w_router, bias, routed_scale)

            # Expert selection is runtime data: the indices drive a gather of the
            # expert weights and a batched matmul over [tokens, top_k]. No static
            # 256-way expansion and no Python control flow.
            g_sel = tf.gather(w_gate, indices, axis=0)
            u_sel = tf.gather(w_up, indices, axis=0)
            d_sel = tf.gather(w_down, indices, axis=0)
            tok4 = tf.reshape(tokens, new_shape=(S, 1, config.hidden_size, 1))
            gate = tf.reshape(tf.matmul(g_sel, tok4), new_shape=(S, _TOPK, _MI))
            up = tf.reshape(tf.matmul(u_sel, tok4), new_shape=(S, _TOPK, _MI))
            h = tf.silu(gate) * up
            h4 = tf.reshape(h, new_shape=(S, _TOPK, _MI, 1))
            down = tf.reshape(tf.matmul(d_sel, h4), new_shape=(S, _TOPK, config.hidden_size))
            routed = tf.reduce(
                down * tf.reshape(weights, new_shape=(S, _TOPK, 1)),
                axes=(1,),
                keepdim=False,
                kind="sum",
            )
            shared = shared_expert(tokens, sh_gate, sh_up, sh_down)
            return tf.reshape(routed + shared, new_shape=(1, S, config.hidden_size))

    @module(target=CudaTarget("nvidia.h200_sxm"))
    class KimiLinear48BA3B:
        """The three kernels this model is, as one tree.

        A root rather than three published submodules: each is its own execution
        domain -- an HIR Function may only call a Function its own Module owns --
        and one root is what lets a caller name any of them by the path it was
        reached through.
        """

        topologies = (Topology("cta", 132),)

        kda = KimiKda
        mla = KimiMla
        moe = KimiMoe

    return KimiLinear48BA3B


#: The published model, at the checkpoint's own config. Any other is the caller's
#: to name -- `reference.py` builds one with fewer experts for the MoE oracle.
KimiLinear48BA3B = build_kimi_linear_48b_a3b(published())

__all__ = ["KimiLinear48BA3B", "build_kimi_linear_48b_a3b"]
