"""MiniCPM3-4B's decode step, made fast: the runtime twin of `model.py`.

`@runtime_module` pins the shape of this file -- the same function names as the
authored Module, the same child tree -- so what is left to decide is what each
of those functions *does*. The answer is one CuTeDSL launch wherever two would
do, six per layer:

    rms_norm + q_a + kv_a  ->  q_b + kv_b  ->  rope + attend
        ->  o_proj + residual  ->  rms_norm + gate/up + swiglu  ->  down + residual

── What the twin owns that the reference does not ───────────────────────────

**Weight layout.** The authored Module declares each projection `(1, in, out)`,
which is what `matmul` reads. A decode GEMV wants the opposite: one output
element per warp, that element's whole `in`-length row contiguous under it.
`load` transposes every projection back to `(out, in)` -- the layout the
published checkpoint stores anyway -- and concatenates the two that share an
input (`q_a_proj`, `kv_a_proj_with_mqa`) so the pair costs one launch and one
read of the normalised hidden state. Nothing downstream sees it: `check`
compares outputs.

**No allocation in a step.** Every intermediate has one shape, known at load
time, so each layer owns a set of buffers written in place. That is what makes
the step capturable.

**One graph.** A step is 374 launches of a few microseconds each; eagerly that
is 2 ms of CPU before the GPU has done anything. `forward` runs the first two
steps eagerly -- to compile the kernels and settle the addresses -- captures the
third, and replays it thereafter. The two things that change between steps, the
token and the position, are device tensors the graph reads rather than arguments
it froze; so is the context length, which is why the attention kernel takes
`ctx` as an `int32[1]` tensor.

**A cache that does not move.** `init_caches` hands out zero-length views into
one preallocated buffer per layer and `append_cache` grows the view by one,
writing all 62 layers' fresh entries with two copies rather than 124
concatenations. A step never mutates the context it was given -- the slot it
writes is one past the end of that view -- and never reallocates, which is what
keeps a long continuation flat instead of quadratic.
"""
from __future__ import annotations

import torch

import kernels as K
from model import (
    EMBED_SCALE, LOGITS_SCALING, MiniCPM3_4B, MiniCPM3_4B_DecoderLayer,
    _KV_A_PROJ, _LORA_EPS, _NOPE, _QK, _V, config,
)
from tilefoundry.runtime import runtime_func, runtime_module

_H = config.num_attention_heads
_HID = config.hidden_size
_INTER = config.intermediate_size
_VOCAB = config.vocab_size
_L = config.num_hidden_layers
_EPS = config.rms_norm_eps
_Q_RANK = config.q_lora_rank
_KV_RANK = config.kv_lora_rank
_ROPE = config.qk_rope_head_dim
_ATTN_OUT = _H * _V
_DT = config.dtype

#: How far a continuation may run before the preallocated cache runs out of
#: slots. Not a model limit -- `config.max_position_embeddings` is 32768 -- but
#: what this twin reserves, at 62 * 2 * 40 * 160 bytes a position (~800 KB).
MAX_CONTEXT = 4096

#: Steps run eagerly before the graph is captured: one to compile every kernel,
#: one to let the caching allocator settle, then capture on the next.
WARMUP_STEPS = 2


def _row_major(w: torch.Tensor) -> torch.Tensor:
    """A declared `(1, in, out)` projection as the `(out, in)` a GEMV reads."""
    return w.squeeze(0).t().contiguous()


class _Ctx:
    """The context length, as a device tensor a kernel can read.

    A captured graph freezes its kernels' scalar arguments, so the number of
    cached positions cannot be one of them. It lives here instead, written by
    whoever knows it and read inside the attention kernel -- and *not* written
    while a capture is in progress, when the value at record time would be the
    only one the graph ever saw.
    """

    def __init__(self, device) -> None:
        self.buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.value = 0

    def set(self, n: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        if n != self.value:
            self.buf.fill_(n)
            self.value = n


class _CacheWindow:
    """Every layer's context, as a sequence that slices only what is asked for.

    `append_cache` has to hand back one `(k, v)` pair per layer, and building
    124 torch views costs about 300 microseconds of Python per token -- a tenth
    of the step, spent on views the fast path never looks at. It looks at one:
    `caches[0][0].shape[1]`, the length. So the pairs are cut on demand.
    """

    __slots__ = ("_k", "_v", "length")

    def __init__(self, k_all, v_all, length: int) -> None:
        self._k, self._v, self.length = k_all, v_all, length

    def __len__(self) -> int:
        return self._k.shape[0]

    def __getitem__(self, i: int):
        return (self._k[i : i + 1, : self.length], self._v[i : i + 1, : self.length])

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class _Scratch:
    """One layer's step-sized buffers, allocated once and written every step."""

    def __init__(self, device) -> None:
        def buf(n):
            return torch.empty(n, device=device, dtype=_DT)

        self.lora = buf(_Q_RANK + _KV_A_PROJ)   # q_a | kv_c | k_rope, one GEMV
        self.q_up = buf(_H * _QK)
        self.kv_up = buf(_H * (_NOPE + _V))
        self.attn = buf(_ATTN_OUT)
        self.proj = buf(_HID)
        self.h1 = buf(_HID)
        self.act = buf(_INTER)
        self.out = buf(_HID)
        # Rebound by the root onto its per-layer slice of one shared pool, so
        # that appending every layer's entry is two copies rather than 124.
        self.k_new = buf(_H * _QK)
        self.v_new = buf(_H * _V)


# ── one decoder layer ────────────────────────────────────────────────────────

def _attention_block(self, hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache,
                     scale, residual, alpha):
    """MLA, and the scaled residual add if the caller wants it folded in.

    `mla_attention` passes no residual, so it stays comparable to the authored
    kernel of that name; `decoder_layer` passes one, which puts `scale_depth`'s
    add in `o_proj`'s epilogue instead of a kernel of its own.
    """
    s = self.scratch
    b = self._bound
    x = hidden.reshape(_HID)
    self._ctx.set(k_cache.shape[1] if k_cache.dim() == 4 else k_cache.shape[0])

    # input_layernorm and both projections that read it, as one launch:
    # `w_q_a` and `w_kv_a` are the two halves of `_w_qkv_a`.
    K.rmsnorm_gemv(x, b["gamma_in"], self._w_qkv_a, _EPS, out=s.lora)

    K.rmsnorm_gemv_pair(
        s.lora[:_Q_RANK], b["gamma_q_a"], b["w_q_b"],
        s.lora[_Q_RANK:_Q_RANK + _KV_RANK], b["gamma_kv_a"], b["w_kv_b"],
        _LORA_EPS, out1=s.q_up, out2=s.kv_up,
    )

    kbuf, vbuf = self._full_caches(k_cache, v_cache)
    K.rope_attend(
        s.q_up, s.kv_up, s.lora[_Q_RANK + _KV_RANK:], cos_cache, sin_cache,
        pos_ids, kbuf, vbuf, self._ctx.buf, scale,
        _H, _NOPE, _ROPE, _V,
        attn=s.attn, k_new=s.k_new, v_new=s.v_new,
    )

    if residual is None:
        out = K.gemv(s.attn, b["w_o"], out=s.proj, plan="o_proj")
    else:
        out = K.gemv_residual(s.attn, b["w_o"], residual.reshape(_HID), alpha,
                              out=s.h1, plan="o_proj")
    return out, s.k_new, s.v_new


def _mlp_block(self, hidden, residual, alpha):
    s = self.scratch
    b = self._bound
    K.rmsnorm_gemv_swiglu(hidden.reshape(_HID), b["gamma_post"], b["w_gate"],
                          b["w_up"], _EPS, out=s.act)
    if residual is None:
        return K.gemv(s.act, b["w_down"], out=s.proj, plan="down")
    return K.gemv_residual(s.act, b["w_down"], residual.reshape(_HID), alpha,
                           out=s.out, plan="down")


def _full_caches(self, k_cache, v_cache):
    """The whole preallocated cache when the view given is one of ours.

    A graph replays fixed addresses, so the attention kernel is handed the
    buffer and told how much of it counts -- but only when the view really is a
    window onto that buffer. Anything else (a cache `check` invented) is passed
    through as it came.
    """
    if self._kbuf is not None and k_cache.data_ptr() == self._kbuf.data_ptr():
        return self._kbuf, self._vbuf
    return k_cache.reshape(-1, _H, _QK), v_cache.reshape(-1, _H, _V)


@runtime_func
def _input_rms_norm(self, hidden, gamma_in):
    return K.rmsnorm(hidden.reshape(_HID), gamma_in, _EPS).reshape(1, 1, _HID)


@runtime_func
def _mla_attention(
    self, hidden, gamma_in, w_q_a, gamma_q_a, w_q_b, w_kv_a, gamma_kv_a, w_kv_b,
    cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
):
    out, k_new, v_new = _attention_block(
        self, hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale,
        None, None,
    )
    return (out.reshape(1, 1, _HID), k_new.reshape(1, 1, _H, _QK),
            v_new.reshape(1, 1, _H, _V))


@runtime_func
def _mlp(self, hidden, gamma_post, w_gate, w_up, w_down):
    return _mlp_block(self, hidden, None, None).reshape(1, 1, _HID)


@runtime_func
def _decoder_layer(
    self, hidden, gamma_in, w_q_a, gamma_q_a, w_q_b, w_kv_a, gamma_kv_a, w_kv_b,
    cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o, gamma_post,
    w_gate, w_up, w_down, residual_scale,
):
    h1, k_new, v_new = _attention_block(
        self, hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale,
        hidden, residual_scale,
    )
    out = _mlp_block(self, h1, h1, residual_scale)
    return out.reshape(1, 1, _HID), k_new.reshape(1, 1, _H, _QK), v_new.reshape(1, 1, _H, _V)


def _layer_load(self, resource) -> None:
    """Read this layer's weights, then put each in the layout its kernel reads."""
    super(MiniCPM3_4B_DecoderLayer_Runtime, self).load(resource)
    b = self._bound
    for name in ("w_q_b", "w_kv_b", "w_o", "w_gate", "w_up", "w_down"):
        b[name] = _row_major(b[name])
    self._w_qkv_a = torch.cat((_row_major(b["w_q_a"]), _row_major(b["w_kv_a"])), 0)
    b["w_q_a"] = self._w_qkv_a[:_Q_RANK]
    b["w_kv_a"] = self._w_qkv_a[_Q_RANK:]
    device = self._w_qkv_a.device
    self.scratch = _Scratch(device)
    if self._ctx is None:
        self._ctx = _Ctx(device)


MiniCPM3_4B_DecoderLayer_Runtime = runtime_module(MiniCPM3_4B_DecoderLayer)(
    type("MiniCPM3_4B_DecoderLayer_Runtime", (), {
        "__doc__": "One MLA decoder layer as six CuTeDSL launches.",
        "input_rms_norm": _input_rms_norm,
        "mla_attention": _mla_attention,
        "mlp": _mlp,
        "decoder_layer": _decoder_layer,
        "load": _layer_load,
        "_full_caches": _full_caches,
        "_ctx": None,
        "_kbuf": None,
        "_vbuf": None,
    })
)


# ── the stack ────────────────────────────────────────────────────────────────

@runtime_func
def _embed(self, w_embed, token_ids):
    return K.embed_scaled(w_embed, token_ids, EMBED_SCALE,
                          out=self._hidden).reshape(1, 1, _HID)


@runtime_func
def _final_rms_norm(self, hidden, gamma_final):
    return K.rmsnorm(hidden.reshape(_HID), gamma_final, _EPS,
                     out=self._normed).reshape(1, 1, _HID)


@runtime_func
def _lm_head(self, hidden, w_head):
    return K.lm_head_gemv(hidden.reshape(_HID), w_head, LOGITS_SCALING,
                          out=self._logits).reshape(1, _VOCAB)


def _root_load(self, resource) -> None:
    super(MiniCPM3_4B_Runtime, self).load(resource)
    # The head is declared `(hidden, vocab)` for `matmul`; a GEMV wants one
    # vocabulary row -- 2560 contiguous elements -- under each warp.
    self._bound["w_head"] = self._bound["w_head"].t().contiguous()
    device = self._bound["w_head"].device

    self._hidden = torch.empty(_HID, device=device, dtype=_DT)
    self._normed = torch.empty(_HID, device=device, dtype=_DT)
    self._logits = torch.empty(_VOCAB, device=device, dtype=_DT)
    self._tok = torch.zeros(1, device=device, dtype=torch.int64)
    self._pos = torch.zeros(1, device=device, dtype=torch.int32)
    self._ctx = _Ctx(device)

    # Every layer's fresh entry in one pool, so appending is two copies.
    self._k_fresh = torch.empty(_L, _H, _QK, device=device, dtype=_DT)
    self._v_fresh = torch.empty(_L, _H, _V, device=device, dtype=_DT)
    for i, layer in enumerate(self.modules):
        layer.scratch.k_new = self._k_fresh[i].reshape(-1)
        layer.scratch.v_new = self._v_fresh[i].reshape(-1)
        layer._ctx = self._ctx


def _init_caches(self, device=None):
    """Zero-length views into one preallocated buffer per layer.

    The reference allocates a fresh `(1, 0, ...)` pair and grows it by
    concatenation; here the pair is `buffer[:, :0]` and growing it is a slice.
    """
    device = torch.accelerator.current_accelerator() if device is None else device
    if self._k_all is None or self._k_all.device != torch.device(device):
        self._k_all = torch.zeros(_L, MAX_CONTEXT, _H, _QK, device=device, dtype=_DT)
        self._v_all = torch.zeros(_L, MAX_CONTEXT, _H, _V, device=device, dtype=_DT)
        for i, layer in enumerate(self.modules):
            layer._kbuf = self._k_all[i]
            layer._vbuf = self._v_all[i]
    self._graph = None
    self._steps = 0
    return _CacheWindow(self._k_all, self._v_all, 0)


def _append_cache(self, caches, fresh):
    """The next step's context: this step's entry written into the slot after
    the view it was given, and the view grown over it.

    Writing lands outside the tensor the step read, so nothing it was handed is
    mutated -- the same contract the reference keeps by allocating a new one.
    """
    at = caches.length if isinstance(caches, _CacheWindow) else caches[0][0].shape[1]
    if fresh[0][0].data_ptr() == self._k_fresh.data_ptr():
        self._k_all[:, at] = self._k_fresh
        self._v_all[:, at] = self._v_fresh
    else:
        for i, (k_new, v_new) in enumerate(fresh):
            self._k_all[i, at] = k_new.reshape(_H, _QK)
            self._v_all[i, at] = v_new.reshape(_H, _V)
    return _CacheWindow(self._k_all, self._v_all, at + 1)


def _prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
    """The token and positional activations for one decode step -- in the same
    buffers every step, so that a captured graph keeps finding them."""
    device = torch.accelerator.current_accelerator() if device is None else device
    if self._cos is None:
        import model as _model

        self._cos, self._sin = _model._generation_rope(torch.device(device))
        self._scale = torch.full((1, 1, 1, 1), _QK ** -0.5, device=device, dtype=_DT)
        self._residual_scale = torch.full(
            (1, 1, 1), config.scale_depth / _L ** 0.5, device=device, dtype=_DT
        )
    self._tok.copy_(input_ids[step].reshape(1))
    self._pos.fill_(step)
    return (self._tok, self._cos, self._sin, self._pos, self._scale,
            self._residual_scale, caches)


def _traverse(self, token_ids, cos_cache, sin_cache, pos_ids, scale,
              residual_scale, caches):
    hidden = self.embed(token_ids)
    fresh = []
    for layer, (k_cache, v_cache) in zip(self.modules, caches):
        hidden, k_new, v_new = layer(
            hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale,
            residual_scale,
        )
        fresh.append((k_new, v_new))
    return self.lm_head(self.final_rms_norm(hidden)), tuple(fresh)


def _root_forward(self, token_ids, cos_cache, sin_cache, pos_ids, scale,
                  residual_scale, caches):
    """One decode step: this token's row, every layer over it, its logits.

    Replayed from a captured graph once the addresses have settled. The
    condition for that is identity, not a flag: the step is only capturable when
    every input is a buffer this twin owns, which is exactly when it came from
    `prepare_inputs_for_generation` and `init_caches`.
    """
    args = (token_ids, cos_cache, sin_cache, pos_ids, scale, residual_scale, caches)
    self._ctx.set(caches.length if isinstance(caches, _CacheWindow)
                  else caches[0][0].shape[1])
    if not self._capturable(args):
        return _traverse(self, *args)

    if self._graph is not None:
        self._graph.replay()
        return self._logits.reshape(1, _VOCAB), self._graph_fresh

    result = _traverse(self, *args)
    self._steps += 1
    if self._steps <= WARMUP_STEPS:
        return result

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _traverse(self, *args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _traverse(self, *args)
    self._graph = graph
    self._graph_fresh = tuple(
        (self._k_fresh[i].reshape(1, 1, _H, _QK), self._v_fresh[i].reshape(1, 1, _H, _V))
        for i in range(_L)
    )
    self._graph.replay()
    return self._logits.reshape(1, _VOCAB), self._graph_fresh


def _capturable(self, args) -> bool:
    token_ids, cos_cache, sin_cache, pos_ids, _, _, caches = args
    return (
        K.capturable()
        and self._k_all is not None
        and token_ids is self._tok
        and pos_ids is self._pos
        and cos_cache is self._cos
        and sin_cache is self._sin
        and isinstance(caches, _CacheWindow)
        and caches._k is self._k_all
    )


MiniCPM3_4B_Runtime = runtime_module(MiniCPM3_4B)(
    type("MiniCPM3_4B_Runtime", (), {
        "__doc__": "The 62-layer stack and the two scaled ends, as one decode step.",
        "embed": _embed,
        "final_rms_norm": _final_rms_norm,
        "lm_head": _lm_head,
        "load": _root_load,
        "forward": _root_forward,
        "init_caches": _init_caches,
        "append_cache": _append_cache,
        "prepare_inputs_for_generation": _prepare_inputs_for_generation,
        "_capturable": _capturable,
        "_k_all": None, "_v_all": None, "_cos": None, "_sin": None,
        "_graph": None, "_graph_fresh": None, "_steps": 0,
        **{f"layer{i}": MiniCPM3_4B_DecoderLayer_Runtime for i in range(_L)},
    })
)


def load(prepared_dir: str, device: str = "cuda") -> "MiniCPM3_4B_Runtime":
    """The twin, with *prepared_dir*'s weights read onto *device*."""
    from tilefoundry.runtime.resource import SafetensorsResource

    twin = MiniCPM3_4B_Runtime()
    twin.load(SafetensorsResource(prepared_dir, device=device))
    return twin


__all__ = [
    "MAX_CONTEXT", "MiniCPM3_4B_DecoderLayer_Runtime", "MiniCPM3_4B_Runtime", "load",
]
