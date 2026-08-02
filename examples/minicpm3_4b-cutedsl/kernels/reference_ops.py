"""The decode path's operations, written in torch.

This is the interface the CuTeDSL kernels implement -- one function per thing
that becomes one kernel launch -- written once in torch so that a kernel can be
moved across behind any single name while the rest of the path stays here. That
is what makes a disagreement have exactly one suspect.

Everything is flat and out-of-place-into-a-buffer: the authored Module's
`(1, 1, hidden)` activations arrive as `(hidden,)` vectors, every projection
weight is `(out, in)` row-major -- what the checkpoint stores and what a GEMV
wants to read -- and every function takes the destination it writes, because a
step that allocates cannot be captured as a graph.
"""
from __future__ import annotations

import torch

__all__ = [
    "embed_scaled", "gemv", "gemv_residual", "lm_head_gemv", "rmsnorm",
    "rmsnorm_gemv", "rmsnorm_gemv_pair", "rmsnorm_gemv_swiglu", "rope_attend",
]


def _into(out, value):
    if out is None:
        return value
    out.copy_(value)
    return out


def _rms(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """`MiniCPMRMSNorm`: normalise in f32, land in bf16, *then* scale.

    The landing before the learned scale is the published order and is visible
    in the last bit, so it is not an implementation detail to tidy away.
    """
    x32 = x.float()
    var = (x32 * x32).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(x.dtype) * gamma


def rmsnorm(x, gamma, eps, out=None):
    return _into(out, _rms(x, gamma, eps))


def rmsnorm_gemv(x, gamma, w, eps, out=None, plan=None):
    """`rms_norm(x, gamma) @ w.T` -- *w* is `(out, in)`."""
    return _into(out, _rms(x, gamma, eps) @ w.t())


def rmsnorm_gemv_pair(x1, g1, w1, x2, g2, w2, eps, out1=None, out2=None):
    """Two independent `rmsnorm_gemv`s; one launch on the CuTeDSL side."""
    return (rmsnorm_gemv(x1, g1, w1, eps, out1),
            rmsnorm_gemv(x2, g2, w2, eps, out2))


def gemv(x, w, out=None, plan=None):
    return _into(out, x @ w.t())


def gemv_residual(x, w, residual, alpha, out=None, plan=None):
    """`residual + (x @ w.T) * alpha` -- the scale_depth residual add, fused."""
    return _into(out, residual + (x @ w.t()).to(x.dtype) * alpha.reshape(()))


def rmsnorm_gemv_swiglu(x, gamma, w_gate, w_up, eps, out=None):
    """`silu(n @ w_gate.T) * (n @ w_up.T)` for `n = rms_norm(x, gamma)`."""
    n = _rms(x, gamma, eps)
    return _into(out, torch.nn.functional.silu(n @ w_gate.t()) * (n @ w_up.t()))


def embed_scaled(table, token, scale, out=None):
    # `index_select`, not `table[int(token)]`: the token id lives on the device
    # and reading it to index with would synchronise.
    return _into(out, torch.index_select(table, 0, token.reshape(1)).reshape(-1) * scale)


def lm_head_gemv(hidden, w_head, divisor, out=None):
    """`(hidden / logits_scaling) @ w_head.T` -- *w_head* is `(vocab, hidden)`.

    A divide, not a multiply by its reciprocal: `MiniCPM3ForCausalLM` divides,
    and 1/10 is not a bf16 number.
    """
    return _into(out, (hidden / divisor) @ w_head.t())


def rope_attend(q_up, kv_up, k_rope, cos, sin, pos, k_cache, v_cache, ctx, scale,
                heads, nope, rope, v_dim, attn=None, k_new=None, v_new=None):
    """Rotate, assemble, and attend the cache together with this token.

    *k_cache* / *v_cache* are `(capacity, heads, ·)` and only their first *ctx*
    rows are context; *ctx* is a one-element device tensor, because the CuTeDSL
    twin reads it inside the kernel. Returns `(attn, k_new, v_new)` flattened.
    """
    qk = nope + rope
    n = int(ctx.reshape(()))
    p = int(pos.reshape(()))
    half = rope // 2

    def rotate(x):
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    c, s = cos[p], sin[p]
    q = q_up.view(heads, qk)
    kv = kv_up.view(heads, nope + v_dim)
    q_rope_e = q[:, nope:] * c + rotate(q[:, nope:]) * s
    k_rope_e = k_rope * c + rotate(k_rope) * s

    q_full = torch.cat((q[:, :nope], q_rope_e), dim=-1)
    k_full = torch.cat((kv[:, :nope], k_rope_e.reshape(1, rope).expand(heads, rope)),
                       dim=-1).contiguous()
    v_full = kv[:, nope:].contiguous()

    qh = (q_full * scale.reshape(())).float()
    score_new = (qh * k_full.float()).sum(-1)
    if n == 0:
        weighted, total = v_full.float(), torch.ones_like(score_new)
    else:
        kc = k_cache[:n].float().permute(1, 0, 2)
        vc = v_cache[:n].float().permute(1, 0, 2)
        score_ctx = torch.einsum("hd,hcd->hc", qh, kc)
        peak = torch.maximum(score_ctx.amax(-1), score_new)
        p_ctx = torch.exp(score_ctx - peak[:, None])
        p_new = torch.exp(score_new - peak)
        total = p_ctx.sum(-1) + p_new
        weighted = (torch.einsum("hc,hcv->hv", p_ctx, vc)
                    + p_new[:, None] * v_full.float())
    out = (weighted / total[:, None]).reshape(-1).to(q_up.dtype)
    return (_into(attn, out), _into(k_new, k_full.reshape(-1)),
            _into(v_new, v_full.reshape(-1)))
