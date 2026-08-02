"""Plain-torch twins of every `@func` in `model.py`.

Why this file exists
--------------------
`model.py` is the authoritative reference, but it is HIR run by an interpreter:
authoritative and slow. This module is the same arithmetic in ordinary torch, and
it earns its place twice.

1. **Bring-up scaffold.** A tilelang kernel can be dropped in one function at a
   time while everything around it still runs, so "does the stack produce
   plausible logits" is answerable before any kernel exists.
2. **Per-function bisect.** A decode step calls ~250 kernels. When a tilelang
   kernel disagrees with the authored Module, the question is *which* one, and
   the only way to ask it is to hold every other function fixed at a version
   known to agree. That is what these are for -- see `TF_IMPL` in
   `kernels/__init__.py`.

Every function here has the **same name, parameter names, and parameter order**
as the `@func` it mirrors, weights included, and returns exactly what that
`@func` returns. That is the whole interface contract: a caller that can drive one
can drive the other, positionally, with no adapter.

There are 16 of them. `model.py` holds 17 `@func` decorators, but
`residual_add` is declared identically by both layer classes (they are separate
Modules because they are separate kernels, not because the add differs).

Dtype policy
------------
Weights arrive in **either float32 or bfloat16**; activations and outputs are
**always float32**.

The matmuls run at the *weights'* dtype -- the activation is cast down to meet
them -- and the result comes back as f32: `(x.to(w.dtype) @ w).float()`. This is
deliberate, and it is the policy that makes this file useful as a bisect tool
rather than merely correct. A bf16 checkpoint's GEMV *is* a bf16 GEMV; if the
reference silently upcast the weights to f32 it would disagree with every
tilelang kernel by ~1e-3 relative, and that disagreement would swamp the real one
being hunted. Everything that is not a matmul -- RMSNorm scales, the convolution
kernel, `a_log`/`dt_bias`, the embedding row -- is lifted to f32 first, matching
the interpreter's own op evaluators, which reduce in f32 regardless of the
tensor's storage dtype.

With f32 weights this file is expected to agree with the authored Module to
better than 1e-5 relative on every output; the self-test at the bottom asserts
exactly that.

Arithmetic details that are easy to get wrong
---------------------------------------------
* `tf.rms_norm(x, w)` is `x * rsqrt(mean(x**2, -1) + eps) * w`, **flat**: `x*w`,
  not `x*(1+w)`. The published `Qwen3_5MoeRMSNorm` is `x*(1+w)`, and that `1+` is
  already folded into the weights by `model.py`'s converters. Doing it again here
  would be wrong by `1+` on 163 of the 285 gammas, in a way no shape check
  catches. The mean-square is taken in f32 whatever the input dtype.
* `l2_normalise` is `rsqrt(sum(x**2) + 1e-6)` -- the **sum**, not the mean. It is
  not an RMSNorm with a unit scale.
* `delta_step` folds the delta rule's query scale `1/sqrt(head_k_dim)` in itself
  (`_QSCALE` in `model.py`); it is an architecture constant, not a parameter.
* `g = -exp(a_log) * softplus(hidden_norm @ w_in_a + dt_bias)`, so `g < 0` and
  `exp(g)` is a decay in (0, 1).
* RoPE is the **rotate-half** form (HF `rotate_half`): the caches are
  `rotary_dim` wide (`cat(freqs, freqs)`) and the pair that rotates together is
  `(x[i], x[i + rotary_dim//2])`, not `(x[2i], x[2i+1])`. Only the leading
  `rotary_dim` of each `head_dim`-wide head is rotated; the tail passes through
  untouched.
* `routing` softmaxes over **every** expert in f32, then takes the top k, then
  renormalises by the top-k sum -- not a softmax over the selected k.
* `routed_experts` holds `w_gate[e]` as `(intermediate, hidden)` = (out, in), so
  the per-expert product is `w_gate[e] @ token`, weight on the left. `w_down[e]`
  is `(hidden, intermediate)`, likewise.

Where this deviates from the authored body (and why it is the same number)
-------------------------------------------------------------------------
`full_attention` in `model.py` scores the cache and the new token as two separate
groups and merges them with an explicit log-sum-exp against their joint maximum.
That is written here as **one ordinary `torch.softmax` over the concatenated
`[cache..., new]` axis**, which is numerically the same computation:
`torch.softmax` subtracts the row maximum over exactly that concatenation, which
*is* the joint maximum the authored merge takes, and normalises by the same total.
The two-group split exists in the authored version because a kernel streams the
cache and the new token from different places; a torch reference has no such
reason to split, and one softmax is the form a reader can check. The `C == 0`
degenerate case comes out right in both: the authored merge gets `-inf` from the
empty max-reduce so `peak == score_new` and `total == 1`, and the softmax over a
length-1 axis gives 1.0.

The scores themselves are *not* written as a matmul. They are the authored
broadcast-multiply-and-sum, so that this file cannot be perturbed by whatever a
process has done to `torch.backends.cuda.matmul.allow_tf32` -- a TF32 score would
lose ~1e-3 relative and look exactly like a broken kernel.

Shapes are read off the tensors, not hard-coded, so these run at any
`Qwen3_5Config` and not only at `REAL`. The one exception is `routing`: `top_k` is
not recoverable from `routing`'s own arguments, so it comes from the config.
"""
from __future__ import annotations

import os
import sys

import torch

# The sibling `config` module, reachable whether this file is imported as
# `kernels.torch_ref` (project root already on the path) or executed directly by
# path -- `python kernels/torch_ref.py` puts `kernels/` on the path, not its
# parent. Same bootstrap, and same reason, as the one at the top of `model.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import REAL as _CFG  # noqa: E402 -- must follow the sys.path bootstrap

#: `tf.rms_norm`'s epsilon, and the model's. Both are 1e-6; `final_rms_norm` is
#: the one authored call that passes it explicitly, as `config.rms_eps`.
_RMS_EPS = _CFG.rms_eps

#: `_L2_EPS` in `model.py`: the linear-attention library's `l2norm` epsilon.
#: Inside the rsqrt, added to the *sum* of squares.
_L2_EPS = 1e-6

#: `config.top_k`. `routing` returns `(S, top_k)` from arguments that do not
#: mention it, so unlike every other extent here it cannot be read off a tensor.
#: Module-level rather than a parameter so the signature stays identical to the
#: authored one; rebind it to run a different configuration.
TOP_K = _CFG.top_k


# ---------------------------------------------------------------------------
# Shared primitives. Not `@func`s -- they are the pieces the authored bodies
# spell out inline, factored only where spelling them twice would let the two
# copies drift.
# ---------------------------------------------------------------------------


def _mm(x, w):
    """``x @ w`` at *w*'s dtype, result f32. Activation on the left."""
    return torch.matmul(x.to(w.dtype), w).float()


def _wm(w, x):
    """``w @ x`` at *w*'s dtype, result f32. Weight on the left -- which is what
    the per-expert products in `routed_experts` are."""
    return torch.matmul(w, x.to(w.dtype)).float()


def _rms_norm(x, weight, eps: float = _RMS_EPS):
    """`tf.rms_norm`: ``x * rsqrt(mean(x**2, -1) + eps) * weight``.

    Flat in *weight* -- no `1 +`; see the module docstring. Reduced in f32
    whatever *x* and *weight* are stored as, matching the interpreter's own
    RMSNorm evaluator.
    """
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(ms + eps) * weight.float()


def _rotate_half(x):
    """HF `rotate_half`: ``cat(-x[..., h:], x[..., :h])`` with ``h`` half the
    last axis. This is the pairing `(x[i], x[i + h])`, which is why the caches
    are `cat(freqs, freqs)`."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _partial_rope(x, cos_cache, sin_cache, pos_ids):
    """The body both `partial_rope` and `partial_rope_kv` have.

    They are two authored Functions only because a Function's parameter shapes
    are fixed and GQA's query and key head counts differ; the arithmetic is one
    thing and is written once.
    """
    rot_dim = cos_cache.shape[-1]
    rot = x[..., :rot_dim].float()
    tail = x[..., rot_dim:].float()
    pos = pos_ids.reshape(-1).long()
    # (1, S, 1, rot_dim): one row per token, broadcast over batch and head.
    cos = cos_cache[pos].float()[None, :, None, :]
    sin = sin_cache[pos].float()[None, :, None, :]
    turned = rot * cos + _rotate_half(rot) * sin
    return torch.cat((turned, tail), dim=-1)


# ---------------------------------------------------------------------------
# Qwen3_5LinearAttention
# ---------------------------------------------------------------------------


def conv_step(conv_state, entry, conv_w):
    """`Qwen3_5LinearAttention.conv_step`.

    conv_state (1, CONV, KERNEL-1), entry (1, CONV, S=1), conv_w (CONV, KERNEL)
    -> (1, CONV)

    The depthwise causal convolution at one token per step. The window closes on
    this token, so the whole convolution is one multiply against the kernel and
    one reduction over it -- channels never mix, which is what depthwise means
    here and why no matmul appears.
    """
    conv, kernel = conv_w.shape
    window = torch.cat((conv_state.float(), entry.float()), dim=2)
    weighted = window * conv_w.float().reshape(1, conv, kernel)
    summed = weighted.sum(dim=-1)
    return torch.nn.functional.silu(summed)


def l2_normalise(x):
    """`Qwen3_5LinearAttention.l2_normalise`.

    x (1, S, HV, DK) -> same. Per-head L2 normalisation, matching the
    linear-attention library's `l2norm`: rsqrt of the **sum** of squares plus
    eps, not of the mean.
    """
    xf = x.float()
    square_sum = xf.pow(2).sum(dim=-1, keepdim=True)
    return xf * torch.rsqrt(square_sum + _L2_EPS)


def delta_step(recurrent_state, q, k, v, g, beta):
    """`Qwen3_5LinearAttention.delta_step`.

    recurrent_state (1, HV, DK, DV), q/k (1, S, HV, DK), v (1, S, HV, DV),
    g/beta (1, S, HV) -> (read (1, HV, DV), updated (1, HV, DK, DV))

    One token of the gated delta rule. The state is an output because a rank-one
    update has no smaller increment to hand back. `q`'s `1/sqrt(DK)` scale is
    folded in here, as it is in the authored body.
    """
    hv, dk, dv = recurrent_state.shape[1:]
    state = recurrent_state.float()
    qf, kf, vf = q.float(), k.float(), v.float()

    decayed = state * torch.exp(g.float()).reshape(1, hv, 1, 1)
    k_col = kf.reshape(1, hv, dk, 1)
    recalled = (decayed * k_col).sum(dim=-2)                      # (1, HV, DV)
    delta = (vf.reshape(1, hv, dv) - recalled) * beta.float().reshape(1, hv, 1)
    updated = decayed + k_col * delta.reshape(1, hv, 1, dv)

    q_scaled = qf * (dk ** -0.5)
    read = (updated * q_scaled.reshape(1, hv, dk, 1)).sum(dim=-2)  # (1, HV, DV)
    return read, updated


def linear_attention(
    hidden,
    gamma_in,
    w_in_qkv,
    w_in_z,
    w_in_b,
    w_in_a,
    conv_w,
    a_log,
    dt_bias,
    conv_state,
    recurrent_state,
    gamma_gdn,
    w_out,
):
    """`Qwen3_5LinearAttention.linear_attention` -- the module entry.

    -> (out (1, S, H), entry (1, CONV, S), updated (1, HV, DK, DV))

    Fused `input_layernorm` + `Qwen3_5MoeGatedDeltaNet`, no residual: the layer
    owns the residual add. `entry` is this step's own pre-convolution column,
    which the caller slides into the window.
    """
    s = hidden.shape[1]
    conv = conv_w.shape[0]
    val = w_in_z.shape[-1]
    hv = w_in_b.shape[-1]
    dv = gamma_gdn.shape[0]
    dk = recurrent_state.shape[2]
    key = (conv - val) // 2          # q and k share one width; v holds the rest
    hk = key // dk
    v_per_k = hv // hk

    hidden_norm = _rms_norm(hidden, gamma_in)

    entry = _mm(hidden_norm, w_in_qkv).permute(0, 2, 1)            # (1, CONV, S)
    mixed = conv_step(conv_state, entry, conv_w)                   # (1, CONV)

    q_flat = mixed[:, :key]
    k_flat = mixed[:, key : 2 * key]
    v_flat = mixed[:, 2 * key : conv]

    # Every value head reads the key head it shares: the projection produces one
    # key head per group, and the delta rule runs per value head.
    q = l2_normalise(
        torch.repeat_interleave(q_flat.reshape(1, s, hk, dk), v_per_k, dim=2)
    )
    k = l2_normalise(
        torch.repeat_interleave(k_flat.reshape(1, s, hk, dk), v_per_k, dim=2)
    )
    v = v_flat.reshape(1, s, hv, dv)

    beta = torch.sigmoid(_mm(hidden_norm, w_in_b))
    # Negative by construction, so exp(g) is a decay in (0, 1): the state cannot
    # grow without a token asking for it through the rank-one update.
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
        _mm(hidden_norm, w_in_a) + dt_bias.float()
    )

    read, updated = delta_step(recurrent_state, q, k, v, g, beta)

    # The gated output norm: normalise per value head, scale, then gate by a
    # projection of the layer input through silu.
    z = _mm(hidden_norm, w_in_z).reshape(1, hv, dv)
    normed = _rms_norm(read, gamma_gdn)
    gated = normed * torch.nn.functional.silu(z)
    out = _mm(gated.reshape(1, s, val), w_out)
    return out, entry, updated


# ---------------------------------------------------------------------------
# Qwen3_5FullAttention
# ---------------------------------------------------------------------------


def partial_rope(x, cos_cache, sin_cache, pos_ids):
    """`Qwen3_5FullAttention.partial_rope`.

    x (1, S, HQ, D) -> same. Rotates the leading `rotary_dim` of each head and
    concatenates the untouched tail back on.
    """
    return _partial_rope(x, cos_cache, sin_cache, pos_ids)


def partial_rope_kv(x, cos_cache, sin_cache, pos_ids):
    """`Qwen3_5FullAttention.partial_rope_kv`.

    x (1, S, HKV, D) -> same. The same rotation over the key's head count.
    """
    return _partial_rope(x, cos_cache, sin_cache, pos_ids)


def full_attention(
    hidden,
    gamma_in,
    w_qg,
    w_k,
    w_v,
    gamma_q,
    gamma_k,
    cos_cache,
    sin_cache,
    pos_ids,
    k_cache,
    v_cache,
    scale,
    w_o,
):
    """`Qwen3_5FullAttention.full_attention` -- the module entry.

    -> (out (1, S, H), k_rope (1, S, HKV, D), v (1, S, HKV, D))

    Fused `input_layernorm` + `Qwen3_5MoeAttention`, no residual. The returned
    key and value are this token's own, for the caller to append to the cache.
    Works at `C == 0` (the first step of a sequence attends the one position it
    brings itself) as well as `C > 0`.

    The authored body merges a cache score group and a new-token score group with
    an explicit log-sum-exp; this is the equivalent single softmax over the
    concatenation. See the module docstring.
    """
    d = gamma_q.shape[0]
    hq = w_qg.shape[-1] // (2 * d)
    hkv = w_k.shape[-1] // d
    group = hq // hkv
    s = hidden.shape[1]

    hidden_norm = _rms_norm(hidden, gamma_in)

    # One projection, two halves: the query and the output gate. The split is
    # over the last axis of the [heads, 2 * head_dim] view, so gate entry j of
    # head h sits beside query entry j of the same head -- not in a second
    # contiguous block of the flat projection.
    qg = _mm(hidden_norm, w_qg).reshape(1, s, hq, 2 * d)
    q = qg[..., :d]
    gate = qg[..., d : 2 * d]

    q_rope = partial_rope(_rms_norm(q, gamma_q), cos_cache, sin_cache, pos_ids)
    k_rope = partial_rope_kv(
        _rms_norm(_mm(hidden_norm, w_k).reshape(1, s, hkv, d), gamma_k),
        cos_cache,
        sin_cache,
        pos_ids,
    )
    v = _mm(hidden_norm, w_v).reshape(1, s, hkv, d)

    q_s = q_rope * scale.float()

    # Cache and new token, then GQA expansion: every query head sees its group's
    # key/value head. `repeat_interleave` is the ordering the authored body uses
    # and the one the head grouping means -- heads 0..G-1 read kv head 0.
    k_all = torch.cat((k_cache.float(), k_rope), dim=1)      # (1, C+S, HKV, D)
    v_all = torch.cat((v_cache.float(), v), dim=1)
    k_g = torch.repeat_interleave(k_all, group, dim=2).permute(0, 2, 1, 3)
    v_g = torch.repeat_interleave(v_all, group, dim=2).permute(0, 2, 1, 3)

    # Scores as the authored broadcast-multiply-and-sum, deliberately not a
    # matmul: a TF32 matmul would lose ~1e-3 relative here and look like a
    # broken kernel. (1, S, HQ, C+S).
    scores = (q_s.unsqueeze(-2) * k_g.unsqueeze(1)).sum(dim=-1)
    probs = torch.softmax(scores, dim=-1)
    attn = (probs.unsqueeze(-1) * v_g.unsqueeze(1)).sum(dim=-2)   # (1, S, HQ, D)

    # The output gate, then o_proj. Head-major flattening on both sides, so gate
    # entry (h, j) meets attention entry (h, j).
    gated = attn.reshape(1, s, hq * d) * torch.sigmoid(gate.reshape(1, s, hq * d))
    return _mm(gated, w_o), k_rope, v


# ---------------------------------------------------------------------------
# Qwen3_5Router
# ---------------------------------------------------------------------------


def routing(tokens, w_router):
    """`Qwen3_5Router.routing` -- the module entry.

    tokens (S, H), w_router (H, E) -> (weights (S, K) f32, indices (S, K) i64)

    HF `Qwen3_5MoeTopKRouter`: softmax over **every** expert in f32, then the
    top k, then renormalise by the top-k sum. `K` is `TOP_K` -- the only extent
    in this file that is not read off an argument.
    """
    logits = _mm(tokens, w_router)                  # already f32
    probs = torch.softmax(logits, dim=-1)
    top_vals, indices = torch.topk(probs, TOP_K, dim=-1)
    denom = top_vals.sum(dim=-1, keepdim=True)
    return (top_vals / denom).float(), indices.to(torch.int64)


# ---------------------------------------------------------------------------
# Qwen3_5MoE
# ---------------------------------------------------------------------------


def post_norm(hidden, gamma_post):
    """`Qwen3_5MoE.post_norm`.

    hidden (1, S, H), gamma_post (H,) -> (S, H)

    HF `post_attention_layernorm`, fused into the MoE block rather than the
    layer, and its own function because the router reads its output.
    """
    h = gamma_post.shape[0]
    s = hidden.shape[1]
    return _rms_norm(hidden, gamma_post).reshape(s, h)


def routed_experts(tokens, weights, indices, w_gate, w_up, w_down):
    """`Qwen3_5MoE.routed_experts`.

    tokens (S, H), weights (S, K), indices (S, K) i64,
    w_gate/w_up (E, I, H), w_down (E, H, I) -> (S, H)

    The gathers are the point: `indices` is a runtime value, so the three expert
    tensors are indexed by it rather than sliced at a known offset. Each token
    then runs K independent SwiGLU experts, batched over the (token, slot) pair,
    and their outputs are mixed by the routing weights.

    `w_gate[e]` is (out, in), so the product is `w_gate[e] @ token` -- weight on
    the left.
    """
    s, h = tokens.shape
    k = indices.shape[-1]
    i = w_gate.shape[1]

    idx = indices.to(torch.int64)
    gate_w = w_gate[idx]                      # (S, K, I, H)
    up_w = w_up[idx]                          # (S, K, I, H)
    down_w = w_down[idx]                      # (S, K, H, I)

    token_col = tokens.reshape(s, 1, h, 1)
    gate = _wm(gate_w, token_col).reshape(s, k, i)
    up = _wm(up_w, token_col).reshape(s, k, i)
    hidden = torch.nn.functional.silu(gate) * up
    down = _wm(down_w, hidden.reshape(s, k, i, 1)).reshape(s, k, h)
    weighted = down * weights.float().reshape(s, k, 1)
    return weighted.sum(dim=1)


def shared_expert(tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale):
    """`Qwen3_5MoE.shared_expert`.

    tokens (S, H), w_shared_gate/up (H, IS), w_shared_down (IS, H),
    w_shared_scale (H, 1) -> (S, H)

    A dense SwiGLU every token goes through, scaled by the token's own scalar
    gate. The gate is a projection to width one through a sigmoid, so it is
    between 0 and 1 per token and cannot change sign.
    """
    gate = _mm(tokens, w_shared_gate)
    up = _mm(tokens, w_shared_up)
    dense = _mm(torch.nn.functional.silu(gate) * up, w_shared_down)
    scale = torch.sigmoid(_mm(tokens, w_shared_scale))
    return dense * scale


def experts(
    tokens,
    weights,
    indices,
    w_gate,
    w_up,
    w_down,
    w_shared_gate,
    w_shared_up,
    w_shared_down,
    w_shared_scale,
):
    """`Qwen3_5MoE.experts` -- the module entry.

    -> (1, S, H)

    `Qwen3_5MoeSparseMoeBlock` once the selection is made, and everything in the
    block that is heavy: the routed experts, the dense shared one, and their mix.
    No residual -- the layer owns the residual add.
    """
    s, h = tokens.shape
    routed = routed_experts(tokens, weights, indices, w_gate, w_up, w_down)
    shared = shared_expert(
        tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
    )
    return (routed + shared).reshape(1, s, h)


# ---------------------------------------------------------------------------
# Qwen3_5FullAttnLayer / Qwen3_5LinearAttnLayer
# ---------------------------------------------------------------------------


def residual_add(a, b):
    """`residual_add`, declared identically by both layer classes.

    a, b (1, S, H) -> (1, S, H)
    """
    return a.float() + b.float()


# ---------------------------------------------------------------------------
# Qwen3_5Decoder
# ---------------------------------------------------------------------------


def embed(table, token_ids):
    """`Qwen3_5Decoder.embed`.

    table (V, H), token_ids (1,) i64 -> (1, S, H)

    HF `Qwen3_5MoeModel.embed_tokens`: the decoded token's own row.
    """
    h = table.shape[-1]
    rows = table[token_ids.to(torch.int64)]
    return rows.float().reshape(1, token_ids.shape[0], h)


def final_rms_norm(hidden, gamma_final):
    """`Qwen3_5Decoder.final_rms_norm`.

    hidden (1, S, H), gamma_final (H,) -> (1, S, H)

    HF `Qwen3_5MoeModel.norm`, applied once after the last layer. The only
    authored `tf.rms_norm` that names its epsilon; it is `config.rms_eps`, the
    same 1e-6 the op defaults to.
    """
    return _rms_norm(hidden, gamma_final, eps=_RMS_EPS)


def lm_head(hidden, w_head):
    """`Qwen3_5Decoder.lm_head`.

    hidden (1, S, H), w_head (H, V) -> (1, V)

    HF `Qwen3_5MoeForCausalLM.lm_head`, over the one token being decoded. Note
    the weight orientation: `model.py`'s converter transposes HF's (V, H) into
    the (H, V) this wants.
    """
    h = w_head.shape[0]
    return _mm(hidden.reshape(1, h), w_head)


__all__ = [
    "TOP_K",
    "conv_step",
    "delta_step",
    "embed",
    "experts",
    "final_rms_norm",
    "full_attention",
    "l2_normalise",
    "linear_attention",
    "lm_head",
    "partial_rope",
    "partial_rope_kv",
    "post_norm",
    "residual_add",
    "routed_experts",
    "routing",
    "shared_expert",
]


# ===========================================================================
# Self-test: every function against the authored Module in `model.py`.
# ===========================================================================

if __name__ == "__main__":
    import math

    import model
    from tilefoundry.runtime import DictResource

    torch.manual_seed(20260730)
    DEV = "cuda"
    CFG = model.config
    TOL = 1e-5

    # ---- input builders -------------------------------------------------
    #
    # Random, but not carelessly so: a projection is scaled by 1/sqrt(fan_in) and
    # a gamma sits near one, because both sides are being compared *relatively*
    # and an input distribution that makes an output nearly cancel reports a
    # large relative error for no reason. The rope caches are the real ones, from
    # `config.rope_theta`.

    def act(*shape, s=1.0):
        return torch.randn(*shape, device=DEV) * s

    def proj(*shape):
        """A weight whose fan-in is its second-to-last axis."""
        return torch.randn(*shape, device=DEV) / math.sqrt(shape[-2])

    def gamma(n):
        return 1.0 + 0.05 * torch.randn(n, device=DEV)

    def rope_caches(rows, rot, theta):
        """`cat(freqs, freqs)` for the rotate-half form: `rot // 2` frequencies,
        each appearing twice, so `(x[i], x[i + rot//2])` rotate together."""
        inv = 1.0 / (
            theta ** (torch.arange(0, rot, 2, device=DEV, dtype=torch.float32) / rot)
        )
        f = torch.outer(torch.arange(rows, device=DEV, dtype=torch.float32), inv)
        emb = torch.cat((f, f), dim=-1)
        return emb.cos(), emb.sin()

    H, V = CFG.hidden, CFG.vocab
    HV, DK, DV = CFG.gdn_n_v_heads, CFG.gdn_head_k_dim, CFG.gdn_head_v_dim
    VAL, CONV, KERNEL, WINDOW = (
        CFG.gdn_value_dim, CFG.gdn_conv_dim,
        CFG.gdn_conv_kernel, CFG.gdn_conv_context,
    )
    HQ, HKV, D, ROT = CFG.n_q_heads, CFG.n_kv_heads, CFG.head_dim, CFG.rotary_dim
    E, K, I, IS = CFG.n_experts, CFG.top_k, CFG.moe_intermediate, CFG.shared_intermediate
    ROWS = CFG.max_ctx

    COS, SIN = rope_caches(ROWS, ROT, CFG.rope_theta)

    def top_k_indices(s=1):
        return torch.stack(
            [torch.randperm(E, device=DEV)[:K] for _ in range(s)]
        ).to(torch.int64)

    def route_weights(s=1):
        w = torch.rand(s, K, device=DEV) + 0.1
        return w / w.sum(dim=-1, keepdim=True)

    # ---- comparison -----------------------------------------------------

    failures: list[str] = []

    def compare(label, names, ours, theirs):
        if not isinstance(ours, tuple):
            ours, theirs = (ours,), (theirs,)
        for name, a, b in zip(names, ours, theirs):
            if a.shape != b.shape:
                line = f"  {label:<22} {name:<10} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
                failures.append(line)
                print(line)
                continue
            if not a.is_floating_point():
                bad = int((a != b).sum())
                mark = "ok" if bad == 0 else "MISMATCH"
                print(f"  {label:<22} {name:<10} exact-int   {bad} differing   {mark}")
                if bad:
                    failures.append(f"{label}/{name}: {bad} indices differ")
                continue
            af, bf = a.float(), b.float()
            denom = bf.abs().max().clamp_min(1e-30)
            rel = ((af - bf).abs().max() / denom).item()
            adiff = (af - bf).abs().max().item()
            mark = "ok" if rel < TOL else "FAIL"
            print(
                f"  {label:<22} {name:<10} max|d|={adiff:9.3e}  "
                f"rel={rel:9.3e}  (|ref|max={bf.abs().max().item():9.3e})  {mark}"
            )
            if not (rel < TOL):
                failures.append(f"{label}/{name}: rel={rel:.3e}")

    # ---- the cases ------------------------------------------------------
    #
    # One builder per authored @func. Each returns (authored, ours, args,
    # output names); the args go to both, positionally, weights included -- that
    # is the whole point of the shared signature. Built lazily so the big weight
    # tensors are freed between cases.

    LA, FA, RT, MO, LY, DC = (
        model.Qwen3_5LinearAttention,
        model.Qwen3_5FullAttention,
        model.Qwen3_5Router,
        model.Qwen3_5MoE,
        model.Qwen3_5LinearAttnLayer,
        model.Qwen3_5Decoder,
    )

    def case_conv_step():
        args = (act(1, CONV, WINDOW), act(1, CONV, 1), act(CONV, KERNEL, s=0.5))
        return LA.conv_step, conv_step, args, ("out",)

    def case_l2_normalise():
        return LA.l2_normalise, l2_normalise, (act(1, 1, HV, DK),), ("out",)

    def case_delta_step():
        q = l2_normalise(act(1, 1, HV, DK))
        k = l2_normalise(act(1, 1, HV, DK))
        g = -torch.rand(1, 1, HV, device=DEV) * 2.0
        beta = torch.sigmoid(act(1, 1, HV))
        args = (act(1, HV, DK, DV, s=0.125), q, k, act(1, 1, HV, DV), g, beta)
        return LA.delta_step, delta_step, args, ("read", "updated")

    def case_linear_attention(zero_state: bool):
        state = (
            (torch.zeros(1, CONV, WINDOW, device=DEV), torch.zeros(1, HV, DK, DV, device=DEV))
            if zero_state
            else (act(1, CONV, WINDOW), act(1, HV, DK, DV, s=0.125))
        )
        args = (
            act(1, 1, H), gamma(H), proj(1, H, CONV), proj(1, H, VAL),
            proj(1, H, HV), proj(1, H, HV), act(CONV, KERNEL, s=0.5),
            act(HV), act(HV), state[0], state[1], gamma(DV), proj(1, VAL, H),
        )
        return LA.linear_attention, linear_attention, args, ("out", "entry", "updated")

    def case_partial_rope():
        pos = torch.randint(0, ROWS, (1,), device=DEV, dtype=torch.int32)
        return FA.partial_rope, partial_rope, (act(1, 1, HQ, D), COS, SIN, pos), ("out",)

    def case_partial_rope_kv():
        pos = torch.randint(0, ROWS, (1,), device=DEV, dtype=torch.int32)
        return (
            FA.partial_rope_kv, partial_rope_kv, (act(1, 1, HKV, D), COS, SIN, pos), ("out",)
        )

    def case_full_attention(C: int):
        pos = torch.full((1,), C, device=DEV, dtype=torch.int32)
        args = (
            act(1, 1, H), gamma(H), proj(1, H, HQ * D * 2), proj(1, H, HKV * D),
            proj(1, H, HKV * D), gamma(D), gamma(D), COS, SIN, pos,
            act(1, C, HKV, D), act(1, C, HKV, D),
            torch.full((1, 1, 1, 1), CFG.attn_scale, device=DEV), proj(1, HQ * D, H),
        )
        return FA.full_attention, full_attention, args, ("out", "k_rope", "v")

    def case_routing():
        return RT.routing, routing, (act(1, H), proj(H, E)), ("weights", "indices")

    def case_post_norm():
        return MO.post_norm, post_norm, (act(1, 1, H), gamma(H)), ("out",)

    def expert_stack():
        """(w_gate, w_up, w_down). `proj` cannot build these: an expert weight is
        (out, in), so its fan-in is the *last* axis, not the second-to-last."""
        return (
            torch.randn(E, I, H, device=DEV) / math.sqrt(H),
            torch.randn(E, I, H, device=DEV) / math.sqrt(H),
            torch.randn(E, H, I, device=DEV) / math.sqrt(I),
        )

    def case_routed_experts():
        args = (act(1, H), route_weights(), top_k_indices(), *expert_stack())
        return MO.routed_experts, routed_experts, args, ("out",)

    def case_shared_expert():
        args = (
            act(1, H), proj(H, IS), proj(H, IS), proj(IS, H),
            torch.randn(H, 1, device=DEV) / math.sqrt(H),
        )
        return MO.shared_expert, shared_expert, args, ("out",)

    def case_experts():
        args = (
            act(1, H), route_weights(), top_k_indices(), *expert_stack(),
            proj(H, IS), proj(H, IS), proj(IS, H),
            torch.randn(H, 1, device=DEV) / math.sqrt(H),
        )
        return MO.experts, experts, args, ("out",)

    def case_residual_add():
        return LY.residual_add, residual_add, (act(1, 1, H), act(1, 1, H)), ("out",)

    def case_embed():
        ids = torch.randint(0, V, (1,), device=DEV, dtype=torch.int64)
        return DC.embed, embed, (act(V, H, s=0.02), ids), ("out",)

    def case_final_rms_norm():
        return DC.final_rms_norm, final_rms_norm, (act(1, 1, H), gamma(H)), ("out",)

    def case_lm_head():
        return DC.lm_head, lm_head, (act(1, 1, H), proj(H, V)), ("out",)

    CASES = [
        ("conv_step", case_conv_step),
        ("l2_normalise", case_l2_normalise),
        ("delta_step", case_delta_step),
        ("linear_attention", lambda: case_linear_attention(False)),
        ("linear_attention C=0", lambda: case_linear_attention(True)),
        ("partial_rope", case_partial_rope),
        ("partial_rope_kv", case_partial_rope_kv),
        ("full_attention C=0", lambda: case_full_attention(0)),
        ("full_attention C=1", lambda: case_full_attention(1)),
        ("full_attention C=37", lambda: case_full_attention(37)),
        ("routing", case_routing),
        ("post_norm", case_post_norm),
        ("routed_experts", case_routed_experts),
        ("shared_expert", case_shared_expert),
        ("experts", case_experts),
        ("residual_add", case_residual_add),
        ("embed", case_embed),
        ("final_rms_norm", case_final_rms_norm),
        ("lm_head", case_lm_head),
    ]

    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}, "
          f"tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"config REAL: hidden={H} vocab={V} experts={E} top_k={K} "
          f"heads q/kv={HQ}/{HKV} head_dim={D} rot={ROT}")
    print(f"f32 weights throughout; tolerance rel < {TOL:g}\n")
    print("== authored @func (positional face: every declared parameter) "
          "vs kernels.torch_ref ==")

    for label, build_case in CASES:
        authored, mine, args, names = build_case()
        ref = authored(*args)
        got = mine(*args)
        compare(label, names, got, ref)
        del authored, mine, args, ref, got
        torch.cuda.empty_cache()

    # ---- the loaded face, and a nested resource --------------------------
    #
    # The same three module entries again, this time through
    # `Mod.load(DictResource(...))` and called with activations alone. Two things
    # are being checked that the positional face cannot show: that a loading
    # binds the ConstTensor params this file names the same way, and that a
    # nested Module's resource really does have to answer `router.<weight>` --
    # `Qwen3_5MoE` has a child `router`, and `load` walks into
    # `resource.subtree("router")` whether or not the entry being run reads it.

    print("\n== loaded face (Mod.load(DictResource(...)), activations only) ==")

    def split(module, fn_name, args):
        """(activations, {weight name: tensor}) for one authored signature.

        Driven by the IR's own `is_const` flags rather than by
        `Module.weights`: the latter is the union over *every* function the
        Module declares, so it names weights the function being run does not
        take (`gamma_post` is `post_norm`'s, not `experts`').
        """
        params = module.lookup(fn_name).params
        acts = [t for p, t in zip(params, args) if not p.is_const]
        weights = {p.name: t for p, t in zip(params, args) if p.is_const}
        return acts, weights

    la_args = case_linear_attention(False)[2]
    la_acts, la_w = split(LA, "linear_attention", la_args)
    la_loaded = LA.load(DictResource(la_w))
    compare(
        "linear_attention", ("out", "entry", "updated"),
        linear_attention(*la_args), la_loaded.linear_attention(*la_acts),
    )
    del la_loaded, la_args, la_w, la_acts
    torch.cuda.empty_cache()

    fa_args = case_full_attention(11)[2]
    fa_acts, fa_w = split(FA, "full_attention", fa_args)
    fa_loaded = FA.load(DictResource(fa_w))
    compare(
        "full_attention C=11", ("out", "k_rope", "v"),
        full_attention(*fa_args), fa_loaded.full_attention(*fa_acts),
    )
    del fa_loaded, fa_args, fa_w, fa_acts
    torch.cuda.empty_cache()

    mo_args = case_experts()[2]
    mo_acts, mo_w = split(MO, "experts", mo_args)
    # `load` binds every weight the *Module* declares, not only the ones the
    # entry being run reads, and it walks into every child. So a resource built
    # to run `experts` still has to answer `gamma_post` (which only `post_norm`
    # reads) and `router.w_router` -- dot-prefixed, because `DictResource` is one
    # flat dict and `subtree("router")` only extends the prefix. Leaving either
    # out is a `KeyError` from `load`, before anything runs.
    mo_w["gamma_post"] = gamma(H)
    mo_w["router.w_router"] = proj(H, E)
    mo_loaded = MO.load(DictResource(mo_w))
    compare(
        "experts", ("out",), experts(*mo_args), mo_loaded.experts(*mo_acts),
    )
    # And the child, loaded and run on its own.
    compare(
        "router.routing", ("weights", "indices"),
        routing(mo_acts[0], mo_w["router.w_router"]),
        mo_loaded.router.routing(mo_acts[0]),
    )
    del mo_loaded, mo_args, mo_w, mo_acts
    torch.cuda.empty_cache()

    # ---- verdict --------------------------------------------------------

    print()
    if failures:
        print(f"FAILED: {len(failures)} comparison(s) over {TOL:g}")
        for line in failures:
            print(f"  {line}")
        raise SystemExit(1)
    print(
        f"OK: every output of all {len(CASES)} cases + 4 loaded-face checks "
        f"agrees with the authored Module to better than {TOL:g} relative."
    )
    print(
        "No function needed a reduced vocab or expert count: the interpreter "
        "runs lm_head (H=2048, V=248320) and routed_experts (E=256) at REAL in "
        "under a second each, so both were compared against the authored @func "
        "itself rather than a hand-written identity."
    )
