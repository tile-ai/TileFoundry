"""`reference_ops`' interface, answered by the CuTeDSL kernels.

Nothing here computes: each function picks the compile key -- the shapes and
scalars a kernel is specialised on -- allocates the destination if the caller
did not bring one, and launches. The keys are what `_common.Compiled` caches on,
so a decode step compiles seven kernels on its first token and none after that.
"""
from __future__ import annotations

import torch

from . import cute_attention as A
from . import cute_gemv as G
from . import cute_misc as M

__all__ = [
    "embed_scaled", "gemv", "gemv_residual", "lm_head_gemv", "rmsnorm",
    "rmsnorm_gemv", "rmsnorm_gemv_pair", "rmsnorm_gemv_swiglu", "rope_attend",
]

#: `(warps, ksplit)` per projection, from tools/tune_gemv.py: every candidate
#: swept at the real shape, timed inside a graph with the weight read cold, and
#: checked against torch before the number was believed. `down` and `q_b|kv_b`
#: keep `ksplit = 1` because the split lost there; the head keeps four warps.
_PLAN = {
    "qkv_a":  (8, 2),
    "qkv_b":  (8, 1),
    "o_proj": (8, 2),
    "swiglu": (8, 2),
    "down":   (20, 1),
    "head":   (4, 1),
}

_ONE = {}


def _like(reference: torch.Tensor, n: int, out):
    if out is not None:
        return out
    return torch.empty(n, device=reference.device, dtype=reference.dtype)


def _dummy(reference: torch.Tensor, n: int) -> torch.Tensor:
    """A stand-in for a tensor the kernel will not read.

    A zero-length KV cache has no address to hand a kernel, and the first step
    of a sequence has exactly that. `ctx` is zero there, so the loop never runs;
    what the pointer points at is immaterial, only that it exists.
    """
    key = (reference.device, reference.dtype, n)
    buf = _ONE.get(key)
    if buf is None:
        buf = torch.zeros(n, device=reference.device, dtype=reference.dtype)
        _ONE[key] = buf
    return buf


def rmsnorm(x, gamma, eps, out=None):
    out = _like(x, x.numel(), out)
    M.RMSNORM((x.numel(), eps), out, x, gamma)
    return out


def rmsnorm_gemv(x, gamma, w, eps, out=None, plan="qkv_a"):
    out = _like(x, w.shape[0], out)
    G.NORM_GEMV((w.shape[1], w.shape[0], eps, *_PLAN[plan]), out, x, gamma, w)
    return out


def rmsnorm_gemv_pair(x1, g1, w1, x2, g2, w2, eps, out1=None, out2=None):
    out1 = _like(x1, w1.shape[0], out1)
    out2 = _like(x2, w2.shape[0], out2)
    G.NORM_GEMV_PAIR((w1.shape[1], w1.shape[0], w2.shape[1], w2.shape[0], eps,
                      *_PLAN["qkv_b"]), out1, x1, g1, w1, out2, x2, g2, w2)
    return out1, out2


def gemv(x, w, out=None, plan="o_proj"):
    out = _like(x, w.shape[0], out)
    G.GEMV((w.shape[1], w.shape[0], False, 1.0, *_PLAN[plan]), out, x, w, out, out)
    return out


def gemv_residual(x, w, residual, alpha, out=None, plan="o_proj"):
    out = _like(x, w.shape[0], out)
    G.GEMV((w.shape[1], w.shape[0], True, 1.0, *_PLAN[plan]), out, x, w, residual,
           alpha.reshape(1))
    return out


def rmsnorm_gemv_swiglu(x, gamma, w_gate, w_up, eps, out=None):
    out = _like(x, w_gate.shape[0], out)
    G.NORM_GEMV_SWIGLU((w_gate.shape[1], w_gate.shape[0], eps, *_PLAN["swiglu"]),
                       out, x, gamma, w_gate, w_up)
    return out


def embed_scaled(table, token, scale, out=None):
    out = _like(table, table.shape[1], out)
    M.EMBED((table.shape[1], float(scale)), out, table, token.reshape(1))
    return out


def lm_head_gemv(hidden, w_head, divisor, out=None):
    out = _like(hidden, w_head.shape[0], out)
    G.LM_HEAD((w_head.shape[1], w_head.shape[0], float(divisor), *_PLAN["head"]),
              out, hidden, w_head)
    return out


def rope_attend(q_up, kv_up, k_rope, cos, sin, pos, k_cache, v_cache, ctx, scale,
                heads, nope, rope, v_dim, attn=None, k_new=None, v_new=None):
    qk = nope + rope
    attn = _like(q_up, heads * v_dim, attn)
    k_new = _like(q_up, heads * qk, k_new)
    v_new = _like(q_up, heads * v_dim, v_new)
    kc = k_cache.reshape(-1) if k_cache.numel() else _dummy(q_up, heads * qk)
    vc = v_cache.reshape(-1) if v_cache.numel() else _dummy(q_up, heads * v_dim)
    A.ROPE_ATTEND(
        (heads, qk, nope, rope, v_dim),
        attn, k_new, v_new, q_up, kv_up, k_rope, cos, sin, pos.reshape(1),
        kc, vc, ctx.reshape(1), scale.reshape(1),
    )
    return attn, k_new, v_new
