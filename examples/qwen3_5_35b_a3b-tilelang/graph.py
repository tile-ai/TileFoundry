"""One decode step, captured into a CUDA graph.

Why this file exists at all is a measurement: an eager tilelang call costs
**20.5 us of pure Python** (a no-op kernel, timed 2000 times), against **0.91 us**
for the same launch replayed from a graph. A step of this model is ~500 launches,
so eager spends ~10 ms a token of which ~95% is Python, and graphed spends
~0.45 ms of launch on top of the ~1.3 ms the weights actually cost. Nothing else
available is worth a factor of five.

What a graph costs in return is that **shapes and addresses are frozen**. Two
things in `model.py`'s contract are not:

1. `full_attention` takes `k_cache: (1, ctx_len, 2, 256)`, whose extent grows by
   one every step, and `append_cache` grows it with `torch.cat`.
2. `Qwen3_5Decoder.forward` is functional: a mixer hands its fresh state back and
   the caller joins it on. The joined tensor is a *new* tensor, so a replay would
   read the state the capture saw, not the state the previous replay wrote.

So this file is the fused, rewritten copy the optimize tutorial's last step
describes -- "Fusion changes the boundaries, so it invalidates the reference you
were using; the way through is a new copy whose reference is the old one." The
new boundaries are:

* the KV cache is one fixed-capacity buffer per layer, and attention masks
  positions past the current one instead of being handed a shorter view. That
  needs a kernel the authored contract has no function for
  (`kernels/attn.full_attention_fixed`), which writes this token's K/V into slot
  `pos` itself.
* the convolution window and the recurrent matrix are persistent buffers that the
  step writes back into, rather than values it returns.

Both are departures from runtime §1.1.2's "a step MUST NOT mutate one it was
given". They are deliberate and they are why `Session.step` is kept: it is the
contract, this is the speed, and `run.py --compare` / `tools/agree.py` diff the
two token streams rather than assuming they match.
"""
from __future__ import annotations

import os

import torch

from kernels import torch_ref as _tr


def _fixed_attention():
    """`full_attention_fixed`: the tilelang one, or a torch stand-in.

    Same `TF_IMPL` switch as everywhere else, and for the same reason -- this is
    the one kernel with no counterpart in the authored Module, so it is the one
    with no `tilefoundry check` to fall back on. Being able to run the graphed
    path on a readable implementation is how its *shape* (cache slots, masking,
    the device-side position) gets separated from its *arithmetic*.
    """
    spec = os.environ.get("TF_IMPL", "")
    if spec == "torch" or "full_attention_fixed:torch" in spec:
        return _torch_fixed
    try:
        from kernels import attn as _attn  # noqa: PLC0415

        return _attn.full_attention_fixed
    except (ImportError, AttributeError):
        return _torch_fixed


def _torch_fixed(
    hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache, sin_cache,
    k_cache, v_cache, pos, scale, w_o, out,
):
    """The fixed-capacity step in torch: correct, graph-safe, not fast.

    Everything that depends on `pos` is a device-side op -- `index_copy_` for the
    write, an `arange <= pos` comparison for the mask -- so the whole thing
    captures. The mask is a `where` on the scores rather than a slice, because a
    slice length would be a host value and there is no host in a replay.
    """
    cap = k_cache.shape[1]
    hq, hkv, d = w_qg.shape[2] // (2 * gamma_q.shape[0]), k_cache.shape[2], gamma_q.shape[0]
    group = hq // hkv

    normed = _tr._rms_norm(hidden, gamma_in)
    qg = (normed.to(w_qg.dtype) @ w_qg[0]).float().view(1, 1, hq, 2 * d)
    q, gate = qg[..., :d], qg[..., d:]
    pos_i32 = pos.to(torch.int32)
    q = _tr._partial_rope(_tr._rms_norm(q, gamma_q), cos_cache, sin_cache, pos_i32)
    k_new = _tr._partial_rope(
        _tr._rms_norm(
            (normed.to(w_k.dtype) @ w_k[0]).float().view(1, 1, hkv, d), gamma_k
        ),
        cos_cache, sin_cache, pos_i32,
    )
    v_new = (normed.to(w_v.dtype) @ w_v[0]).float().view(1, 1, hkv, d)

    slot = pos.to(torch.int64)
    k_cache.index_copy_(1, slot, k_new)
    v_cache.index_copy_(1, slot, v_new)

    # (hq, cap) scores against the whole buffer, then mask the slots past `pos`.
    keys = k_cache[0].repeat_interleave(group, dim=1)          # (cap, hq, d)
    vals = v_cache[0].repeat_interleave(group, dim=1)
    scores = ((q[0, 0] * scale.reshape(())) .unsqueeze(0) * keys).sum(-1)  # (cap, hq)
    live = (torch.arange(cap, device=pos.device) <= pos).unsqueeze(1)
    scores = torch.where(live, scores, torch.full_like(scores, float("-inf")))
    probs = torch.softmax(scores, dim=0)
    attn = (probs.unsqueeze(-1) * vals).sum(0)                 # (hq, d)

    gated = attn.reshape(1, 1, hq * d) * torch.sigmoid(gate.reshape(1, 1, hq * d))
    out.copy_((gated.to(w_o.dtype) @ w_o[0]).float())


def _bound(node):
    """A twin's loaded weights.

    `_Twin` keeps them in `_bound` and exposes them only by filling a kernel's
    `ConstTensor` parameters at call time (runtime §1.1). This file calls a
    kernel the authored Module does not declare, so it has to reach them
    directly; there is no public face for "the weights of this node" because
    under the contract nobody needs one.
    """
    return node._bound


class GraphedStep:
    """A `Session`'s decode step, replayable.

    Holds every buffer the step reads or writes, so a replay is a pure function
    of `token` and `pos` -- both of which are device tensors the host overwrites
    between replays.
    """

    def __init__(self, session) -> None:
        self.session = session
        self.cfg = session.cfg
        self.device = session.device
        self.capacity = session.capacity
        self.twin = session.twin
        cfg = self.cfg

        f32 = torch.float32
        dev = self.device

        # -- what the host writes between replays -------------------------
        self.token = torch.zeros(1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)

        # -- per-layer state, allocated once ------------------------------
        self.conv = {}
        self.recur = {}
        self.kbuf = {}
        self.vbuf = {}
        for index, kind in enumerate(cfg.layer_types):
            if kind == "linear_attention":
                self.conv[index] = torch.zeros(
                    1, cfg.gdn_conv_dim, cfg.gdn_conv_context, dtype=f32, device=dev
                )
                self.recur[index] = torch.zeros(
                    1, cfg.gdn_n_v_heads, cfg.gdn_head_k_dim, cfg.gdn_head_v_dim,
                    dtype=f32, device=dev,
                )
            else:
                self.kbuf[index] = torch.zeros(
                    1, self.capacity, cfg.n_kv_heads, cfg.head_dim, dtype=f32, device=dev
                )
                self.vbuf[index] = torch.zeros(
                    1, self.capacity, cfg.n_kv_heads, cfg.head_dim, dtype=f32, device=dev
                )

        self.cos, self.sin = session.cos, session.sin
        self.scale = session.scale
        self.logits = torch.zeros(1, cfg.vocab, dtype=f32, device=dev)
        self._fixed = _fixed_attention()
        self._graph = None

    # -- state -----------------------------------------------------------

    def reset(self) -> None:
        for buf in (*self.conv.values(), *self.recur.values(),
                    *self.kbuf.values(), *self.vbuf.values()):
            buf.zero_()
        self.pos.zero_()
        self._n = 0

    # -- the step, written out once ---------------------------------------

    def _body(self):
        """One decode step against the persistent buffers.

        This is `Qwen3_5Decoder.forward` and `_layer_forward` and
        `Qwen3_5MoE.forward` inlined into one function -- not because inlining is
        faster (it is the same kernels in the same order) but because the state
        handling has to change, and the authored orchestration methods are reused
        verbatim on the twin by contract, so they cannot be the place it changes.
        """
        twin = self.twin
        hidden = twin.embed(self.token)

        for index, kind in enumerate(self.cfg.layer_types):
            layer = getattr(twin, f"layer{index}")
            if kind == "linear_attention":
                mixed, entry, updated = layer.mixer.linear_attention(
                    hidden, self.conv[index], self.recur[index]
                )
                # The window slides by the one column this step produced. The
                # `cat` allocates from the graph's private pool, which is what
                # makes the read-then-write of one buffer safe.
                self.conv[index].copy_(
                    torch.cat([self.conv[index][:, :, 1:], entry], dim=2)
                )
                self.recur[index].copy_(updated)
            else:
                mixed = torch.empty_like(hidden)
                w = _bound(layer.mixer)
                self._fixed(
                    hidden, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                    w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                    self.kbuf[index], self.vbuf[index], self.pos,
                    self.scale, w["w_o"], mixed,
                )
            attended = layer.residual_add(hidden, mixed)
            tokens = layer.moe.post_norm(attended)
            weights, indices = layer.moe.router.routing(tokens)
            expert_out = layer.moe.experts(tokens, weights, indices)
            hidden = layer.residual_add(attended, expert_out)

        normed = twin.final_rms_norm(hidden)
        logits = twin.lm_head(normed)
        # Advancing the position is part of the step, so it belongs inside the
        # capture: left outside it would be one more eager launch per token, and
        # eager launches are the whole reason this file exists.
        self.pos.add_(1)
        return logits

    def build(self) -> None:
        """Warm up, then capture.

        The warmup is on a side stream because that is what `torch.cuda.graph`
        requires, and it is where every tilelang kernel gets compiled -- ~40
        distinct kernels, seconds each, once per process.
        """
        self.reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(side)
        self.reset()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            out = self._body()
        # `out` is allocated inside the capture, so its address is fixed and the
        # same tensor carries every replay's result.
        self._out = out
        self.reset()

    # -- the driver face, same as `Session` -------------------------------

    def step(self, token_id: int):
        self.token.fill_(token_id)
        if self._graph is None:
            return self._body()
        self._graph.replay()
        return self._out


__all__ = ["GraphedStep"]
