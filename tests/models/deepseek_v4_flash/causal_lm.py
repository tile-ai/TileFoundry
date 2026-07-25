"""DeepSeek-V4-Flash causal-LM model tree: the two levels above the verified
leaf components (``attention.py``'s ``build_attention``, ``moe.py``'s
``build_moe`` -- the hash-router variant, real layers ``0..num_hash_layers-1``
per config.json).

    DeepseekV4ForCausalLM (root)          embed / final_rms_norm / lm_head
    └─ layer0 .. layerN-1                 pre_attn_rms_norm / pre_moe_rms_norm
       (DeepseekV4DecoderLayer)           / residual_add
       ├─ attention = build_attention(config)
       └─ moe        = build_moe(config)

Replaces ``decode_step.py`` (deleted): that file was "HIR structural fixture
only (no execution or orchestration lives here)"; this one adds the
orchestration methods (``forward`` / ``init_caches`` /
``prepare_inputs_for_generation``) that ``tilefoundry.runtime.generation.
generate`` needs, and is exercised end to end (``prepare`` -> ``load`` ->
``generate``), not just structurally.

Two things ``decode_step.py`` got away with that a real ``prepare()`` run
does not:

- ``final_rms_norm`` and ``lm_head`` both declared their own weight as a
  ConstTensor named plain ``"weight"``. ``Module.weights`` unions ConstTensor
  params *by name* across every function of a node (raising if two functions
  give the same name different ``TensorType``s -- ``ir/core/module.py``), and
  the two really are different shapes (``(dim,)`` vs ``(dim, vocab)``); this
  was never exercised (``decode_step_module.weights`` / ``.prepare`` was never
  called there). Renamed here to ``final_norm_weight`` / ``lm_head_weight``.
- ``lm_head``'s weight was declared ``(dim, vocab)`` with no converter, i.e.
  assumed to load byte-identical from the checkpoint. The real ``head.weight``
  is ``(vocab, dim)`` (a plain ``nn.Linear``-style orientation, like
  ``embed.weight`` -- but unlike ``embed``, whose gather-by-row use needs
  exactly that orientation, ``lm_head``'s own ``matmul`` convention needs the
  transpose). Fixed here with a plain-transpose ``.converter(...)``, the same
  shape as ``attention.py``'s unscaled ``w_o_a`` converter.

Layer replication: ``config.n_layers`` copies of the *same* architecture
(real layer 0's -- pure sliding-window MLA + hash-router MoE), per this
task's brief, regardless of what the real checkpoint's other layers actually
look like (``compress_ratios`` varies per real layer; only real layer 0 --
and the hash router, real layers 0..2 -- are verified components here).
Built as ``tuple(build_decoder_layer(config).renamed(f"layer{i}") for i in
range(config.n_layers))`` -- the factory call is *inside* the comprehension,
deliberately not hoisted out to one shared instance renamed N times:
``Module.renamed`` is a shallow ``dataclasses.replace(name=...)``
(``ir/core/module.py``), so a shared instance's nested ``attention`` / ``moe``
children would be the *same* Python objects across every layer, and
``Module.load`` binds each node's converted weights onto that node via
``object.__setattr__(self, "_bound", ...)`` -- i.e. two layers sharing one
``attention`` object would have the second ``load()`` silently clobber the
first's weights. Calling the factory fresh per index gives every layer (and
every layer's own ``attention`` / ``moe``) an independent object graph, so
each keeps its own ``_bound`` state. Immaterial at this task's own gate
(``DSV4Config.tiny()`` has ``n_layers == 1``) but a real correctness
requirement at ``REAL`` (43 layers) -- so no eager ``causal_lm_module =
build_causal_lm(REAL)`` module-level binding is exported here (unlike
``attention.py`` / ``moe.py``'s eager ``REAL``-scale bindings): a quick
timing showed ~0.27s per fresh (``build_attention`` + ``build_moe``) pair, so
43 of them at import time would add roughly a dozen seconds to every process
that imports this module -- callers needing the real scale should call
``build_causal_lm(REAL)`` themselves.

Where the pre-FFN norm lives: the real checkpoint's per-layer
``ffn_norm.weight`` is a LAYER-level tensor (a sibling of ``ffn``, not a
tensor inside it), so it is ``DeepseekV4DecoderLayer`` that owns
``pre_moe_rms_norm`` here (real weight, aliased in ``hf_alias.py``) and
applies it before calling ``self.moe(...)`` -- the checkpoint- and
textbook-faithful ``h = h + ffn(ffn_norm(h))``. The hash MoE component
accordingly has no internal pre-norm of its own, so the forward path
normalizes exactly once. (``build_moe_learned`` does still carry one, against
a component-local ``rms_weight`` with no checkpoint backing; that asymmetry
is deliberate and documented in ``moe.py`` -- the learned variant is used
only as a standalone planning problem by ``tests/schedule/*``, never nested
into this tree.) The attention side never had the question: ``attention.py``
has no internal pre-attention norm at all.

RoPE: ``DSV4Config`` (``config.py``, not ours to edit) carries no
``rope_theta`` / ``rope_scaling`` fields, so ``_ROPE_PARAMS`` below reads them
directly from the same checked-in ``config.json`` that ``config.py`` itself
parses (mirroring its own ``from_config_json`` pattern, just for the fields
its dataclass does not carry) -- fixed architectural constants of the real
checkpoint, used regardless of *config* scale (``DSV4Config.tiny()`` keeps
``rope_dim`` identical to ``REAL``, so the YaRN correction range is identical
too). ``_yarn_inv_freq_and_scale`` reproduces the standard YaRN formula (the
same public algorithm as ``transformers.modeling_rope_utils``'s ``"yarn"``
rope-init function) by hand, since no ``transformers`` ``DeepseekV4`` config
class exists to drive that generic helper directly (see ``config.py``'s own
module docstring) and DeepSeek's MLA yarn dimension is ``qk_rope_head_dim``
(``config.rope_dim``), not the full ``head_dim``. Precision against a real
reference is out of scope for this task (no oracle exists yet -- "a single
end-to-end test lands in the next phase by design"); the gate here is finite
logits, not a numerical match.

Also out of scope, spotted but not modeled: the real checkpoint carries
per-layer ``hc_attn_base`` / ``hc_attn_fn`` / ``hc_attn_scale`` /
``hc_ffn_base`` / ``hc_ffn_fn`` / ``hc_ffn_scale`` tensors (plus a top-level
``hc_head_*`` triple), and config.json has ``hc_mult`` / ``hc_sinkhorn_iters``
/ ``hc_eps`` fields -- some kind of learned "hyper-connections" mixing in
place of a plain residual add. This task's own brief gives the layer's
forward as plain residual adds (``h = x + attn(...)``, ``h = h + ffn(...)``),
which is what is built here; the ``hc_*`` tensors are unused/unaliased.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tests.models.deepseek_v4_flash.attention import build_attention
from tests.models.deepseek_v4_flash.config import DSV4Config
from tests.models.deepseek_v4_flash.moe import build_moe
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.module import Module


def _load_rope_params() -> dict:
    """``rope_theta`` / ``rope_scaling`` (YaRN), read directly from the
    checked-in real config.json -- see module docstring (``DSV4Config``
    carries neither, and config.py is not ours to edit)."""
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    scaling = cfg["rope_scaling"]
    if scaling["type"] != "yarn":
        raise ValueError(f"unsupported rope_scaling type {scaling['type']!r}")
    return {
        "theta": float(cfg["rope_theta"]),
        "factor": float(scaling["factor"]),
        "beta_fast": float(scaling["beta_fast"]),
        "beta_slow": float(scaling["beta_slow"]),
        "orig_max_pos": int(scaling["original_max_position_embeddings"]),
    }


_ROPE_PARAMS = _load_rope_params()


def _yarn_inv_freq_and_scale(rope_dim: int):
    """YaRN inverse frequencies (one per rotated pair, ``rope_dim // 2`` of
    them) plus the post-hoc attention (mscale) scale folded into cos/sin at
    construction -- the standard public YaRN formula (see module docstring).
    """
    import torch  # noqa: PLC0415 -- deferred; only needed at generation time

    theta = _ROPE_PARAMS["theta"]
    factor = _ROPE_PARAMS["factor"]
    beta_fast = _ROPE_PARAMS["beta_fast"]
    beta_slow = _ROPE_PARAMS["beta_slow"]
    orig_max_pos = _ROPE_PARAMS["orig_max_pos"]

    pos_freqs = theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def _correction_dim(num_rotations: float) -> float:
        return (
            rope_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(theta))

    low = max(math.floor(_correction_dim(beta_fast)), 0)
    high = min(math.ceil(_correction_dim(beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001  # guard the degenerate ramp (mirrors the public formula)
    ramp = (torch.arange(rope_dim // 2, dtype=torch.float64) - low) / (high - low)
    extrapolation_factor = 1.0 - ramp.clamp(0, 1)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_factor


def _rope_cos_sin(config: DSV4Config, position: int, *, device):
    """cos / sin for one absolute sequence *position*, each
    ``(config.rope_half,)`` f32 -- one value per rotated pair, matching
    ``attention.py``'s interleaved-pairs convention (see its module
    docstring): pair ``i`` (head dims ``2i``/``2i+1`` of the rope slice)
    rotates by ``position * inv_freq[i]``."""
    import torch  # noqa: PLC0415

    inv_freq, attention_factor = _yarn_inv_freq_and_scale(config.rope_dim)
    angles = position * inv_freq
    cos = (angles.cos() * attention_factor).to(dtype=torch.float32, device=device)
    sin = (angles.sin() * attention_factor).to(dtype=torch.float32, device=device)
    return cos, sin


def _decode_attn_mask(config: DSV4Config, step: int, *, device):
    """Additive mask ``(1, 1, 1, config.window)`` bf16 for decode step *step*
    over the fixed-capacity sliding-window cache. Cache slot ``j`` holds a
    real (already-written) token iff ``j <= step`` while the window is still
    filling (``step < window - 1``); once ``step >= window - 1`` every
    physical slot has been written at least once (the last ``window`` tokens,
    in circular order -- softmax attention is order-invariant over keys), so
    no slot needs masking."""
    import torch  # noqa: PLC0415

    mask = torch.zeros(config.window, dtype=torch.bfloat16, device=device)
    if step < config.window - 1:
        mask[step + 1 :] = float("-inf")
    return mask.view(1, 1, 1, config.window)


def build_decoder_layer(config: DSV4Config) -> Module:
    """One decoder layer: its own two pre-norms + residual add (real,
    layer-level checkpoint tensors -- see module docstring), nesting
    ``build_attention(config)`` / ``build_moe(config)`` under the
    ``attention`` / ``moe`` attributes (``@module``'s own torch/HF renaming
    -- see ``attention.py`` / ``moe.py``'s module docstrings)."""

    @module(entry="residual_add")
    class DeepseekV4DecoderLayer:
        @func
        def pre_attn_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_attn_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, pre_attn_norm_weight)

        @func
        def pre_moe_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_moe_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, pre_moe_norm_weight)

        @func
        def residual_add(
            a: Tensor[(1, 1, config.dim), "bf16"],
            b: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.add(a, b)

        attention = build_attention(config)
        moe = build_moe(config)

        def forward(
            self, hidden, cos_pos, sin_pos, cur_pos, s, kv_cache0, attn_mask, scale,
            ones_head_dim, token_ids,
        ):
            """``h1 = hidden + attn(attn_norm(hidden))``, then ``out = h1 +
            ffn(ffn_norm(h1))`` -- see module docstring for the ffn side's
            flagged double-normalize (``self.moe`` re-normalizes internally,
            unavoidably, with its own checkpoint-less weight)."""
            attn_in = self.pre_attn_rms_norm(hidden)
            attn_out, kv_cache1 = self.attention(
                attn_in, cos_pos, sin_pos, cur_pos, s, kv_cache0, attn_mask, scale, ones_head_dim,
            )
            h1 = self.residual_add(hidden, attn_out)
            moe_in = self.pre_moe_rms_norm(h1)
            moe_out = self.moe(moe_in, token_ids)
            out = self.residual_add(h1, moe_out)
            return out, kv_cache1

    return DeepseekV4DecoderLayer


def build_causal_lm(config: DSV4Config) -> Module:
    """The full model tree: ``embed`` -> ``config.n_layers`` decoder layers
    (threading each their own KV cache) -> ``final_rms_norm`` -> ``lm_head``,
    plus the ``generate()``-facing orchestration hooks (``forward`` /
    ``init_caches`` / ``prepare_inputs_for_generation`` --
    ``tilefoundry.runtime.generation.generate``'s contract)."""

    @module(entry="lm_head")
    class DeepseekV4ForCausalLM:
        @func
        def embed(
            table: ConstTensor[(config.vocab, config.dim), "bf16"],
            token_ids: Tensor[(1,), "i64"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.reshape(tf.gather(table, token_ids, axis=0), new_shape=(1, 1, config.dim))

        @func
        def final_rms_norm(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            final_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(hidden, final_norm_weight)

        @func
        def lm_head(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            lm_head_weight: ConstTensor[(config.dim, config.vocab), "bf16"],
        ) -> Tensor[(1, 1, config.vocab), "bf16"]:
            logits = tf.matmul(tf.reshape(hidden, new_shape=(1, config.dim)), lm_head_weight)
            return tf.reshape(logits, new_shape=(1, 1, config.vocab))

        @lm_head.converter("lm_head_weight")
        def _(
            head_weight_raw: ConstTensor[(config.vocab, config.dim), "bf16"],
        ) -> Tensor[(config.dim, config.vocab), "bf16"]:
            # Real `head.weight` is (vocab, dim) (nn.Linear-style, like
            # `embed.weight`); lm_head's own matmul convention needs the
            # transpose -- see module docstring.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        # One independent tree per layer, built fresh -- see module
        # docstring for why this must not be `build_decoder_layer(config)`
        # (built once) `.renamed(...)` N times.
        layers = tuple(
            build_decoder_layer(config).renamed(f"layer{i}") for i in range(config.n_layers)
        )

        def forward(
            self, token_ids, cos_pos, sin_pos, cur_pos, s, past_key_values, attn_mask, scale,
            ones_head_dim,
        ):
            hidden = self.embed(token_ids)
            new_caches = []
            for i in range(config.n_layers):
                layer = getattr(self, f"layer{i}")
                hidden, new_cache = layer(
                    hidden, cos_pos, sin_pos, cur_pos, s, past_key_values[i], attn_mask, scale,
                    ones_head_dim, token_ids,
                )
                new_caches.append(new_cache)
            normed = self.final_rms_norm(hidden)
            logits = self.lm_head(normed)
            return logits, tuple(new_caches)

        def init_caches(self, device="cuda", mesh=None):
            """One zero KV-cache slot per layer -- fixed
            ``(1, config.window, 1, config.head_dim)`` sliding-window
            capacity, matching ``attention.py``'s ``kv_cache0`` param.
            ``mesh`` is accepted for a future sharded-cache-init hook and is
            currently unused."""
            import torch  # noqa: PLC0415

            return tuple(
                torch.zeros(1, config.window, 1, config.head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(config.n_layers)
            )

        def prepare_inputs_for_generation(self, input_ids, step, past_key_values, device="cuda"):
            """The complete positional argument tuple ``forward`` needs for
            decode step *step*: ``cur_pos`` is the sliding-window cache slot
            (``step % config.window``); RoPE's cos/sin use the *absolute*
            ``step`` instead (see ``_rope_cos_sin`` / module docstring)."""
            import torch  # noqa: PLC0415

            ids = input_ids.reshape(-1)
            token_ids = ids[step].reshape(1).to(device=device, dtype=torch.int64)
            cur_pos = torch.tensor([step % config.window], device=device, dtype=torch.int32)
            s = torch.tensor([1], device=device, dtype=torch.int32)
            cos, sin = _rope_cos_sin(config, step, device=device)
            cos_pos = cos.view(1, 1, 1, config.rope_half)
            sin_pos = sin.view(1, 1, 1, config.rope_half)
            attn_mask = _decode_attn_mask(config, step, device=device)
            scale = torch.full(
                (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=torch.bfloat16,
            )
            ones_head_dim = torch.ones(config.head_dim, device=device, dtype=torch.bfloat16)
            return (
                token_ids, cos_pos, sin_pos, cur_pos, s, past_key_values, attn_mask, scale,
                ones_head_dim,
            )

    return DeepseekV4ForCausalLM


__all__ = ["build_causal_lm", "build_decoder_layer"]
