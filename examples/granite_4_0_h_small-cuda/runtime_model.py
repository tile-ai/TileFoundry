"""The fast implementation of `model.py`: one runtime twin per authored Module,
every kernel body a launch into `kernels/granite.cu`.

`@runtime_module` holds the two sides to the same shape -- same function names,
same children, same entry -- so a twin can be measured against the Module it
stands for one function at a time, which is how every kernel here was arrived
at. The orchestration is not re-written: `forward` on a layer, on the MoE block
and on the root are the authored methods, running over the twin's kernels.

Two departures from the reference are deliberate, and both are what make decode
fast rather than merely correct:

* **State advances in place.** The Mamba convolution window and recurrent
  matrix, and both attention caches, are written by the kernel that reads them
  and handed back as the same tensors. The reference returns fresh values
  because that is what a value semantics has to do; a decode loop that
  reallocated 147 MB of recurrent state every step would spend more time in the
  allocator than in the model. `append_cache` is therefore the identity here.
* **`cur_pos`, not `mask`.** Both say which of the KV capacity is live. The
  kernel reads the number and stops there, so a short context costs a short
  loop. The mask parameter is still declared, because the twin's signatures are
  the Module's.

`generate` on the root is where the two pay off: with every buffer at a fixed
address and every extent static, one decode step captures into a CUDA graph and
each subsequent token is a single launch instead of roughly four hundred.
"""
from __future__ import annotations

from time import perf_counter

import torch

import model as sem
from kernels import ops
from tilefoundry.runtime import runtime_func, runtime_module

_K = ops()

_BF = sem._TORCH_DT
_H = sem._H
_EPS = sem._EPS
_PROJ, _CONVD, _EXP = sem._PROJ, sem._CONVD, sem._EXP
_NH, _PD, _NS, _NG = sem._NH, sem._PD, sem._NS, sem._NG
_WIN = sem._WIN
_HQ, _HKV, _HD = sem._HQ, sem._HKV, sem._HD
_CAP = sem._CAP
_E, _KTOP, _I, _IS = sem._E, sem._K, sem._I, sem._IS
_V = sem._V
#: Must match ATTN_SPLITS in kernels/granite.cu -- the workspace is sized by it.
_ATTN_SPLITS = 8


def _empty(shape, device, dtype=_BF):
    return torch.empty(shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Mamba-2 mixer
# ---------------------------------------------------------------------------


@runtime_module(sem.GraniteMamba)
class MambaMixer:
    """Four kernels: the fused pre-norm and projection, the depthwise
    convolution, the recurrence, and the gated output norm with its
    projection."""

    @runtime_func
    def in_projection(self, hidden, gamma_in, w_in):
        normed = _empty((1, 1, _H), hidden.device)
        _K.rms_norm(hidden, gamma_in, normed, _EPS)
        proj = _empty((1, _PROJ), hidden.device)
        _K.gemv(w_in, normed, proj)
        return proj

    @runtime_func
    def conv_step(self, conv_state, entry, conv_w, conv_b):
        # Advances `conv_state` in place: the window this token closes on is
        # the window the next token opens from, and it is 50 KB that would
        # otherwise be reallocated 35 times a step.
        out = _empty((1, _CONVD), entry.device)
        _K.mamba_conv(conv_state, entry, conv_w, conv_b, out)
        return out

    @runtime_func
    def ssm_step(self, ssm_state, x, b_vec, c_vec, dt_raw, a_log, dt_bias, d_skip):
        # Advances `ssm_state` in place, for the same reason and 80x the bytes.
        y = _empty((1, _NH, _PD), x.device)
        _K.mamba_ssm(ssm_state, x, b_vec, c_vec, dt_raw, a_log, dt_bias, d_skip, y)
        return y, ssm_state

    @runtime_func
    def gated_out(self, y, gate, gamma_ssm, w_out):
        normed = _empty((1, _EXP), y.device)
        _K.rms_norm_gated(y, gate, gamma_ssm, normed, _EPS)
        out = _empty((1, 1, _H), y.device)
        _K.gemv(w_out, normed, out)
        return out

    @runtime_func
    def mamba_mixer(
        self, hidden, gamma_in, w_in, conv_w, conv_b, a_log, dt_bias, d_skip,
        conv_state, ssm_state, gamma_ssm, w_out,
    ):
        # The projection's three consumers are views into it, never copies:
        # the gate, the convolved q/B/C stream, and one dt per head.
        proj = self.in_projection(hidden)
        gate = proj[:, :_EXP]
        entry = proj[:, _EXP : _EXP + _CONVD].reshape(1, _CONVD, 1)
        dt_raw = proj[:, _EXP + _CONVD :]

        mixed = self.conv_step(conv_state, entry)
        x = mixed[:, :_EXP].reshape(1, _NH, _PD)
        b_vec = mixed[:, _EXP : _EXP + _NG * _NS].reshape(1, _NG, _NS)
        c_vec = mixed[:, _EXP + _NG * _NS :].reshape(1, _NG, _NS)

        y, updated = self.ssm_step(ssm_state, x, b_vec, c_vec, dt_raw)
        return self.gated_out(y, gate), entry, updated


# ---------------------------------------------------------------------------
# Full attention
# ---------------------------------------------------------------------------


@runtime_module(sem.GraniteAttention)
class Attention:
    """Three kernels for the projections, one online-softmax pass over the live
    cache, one more for the output projection. No rotary: `nope`."""

    @runtime_func
    def qkv(self, hidden, gamma_in, w_q, w_k, w_v):
        normed = _empty((1, 1, _H), hidden.device)
        _K.rms_norm(hidden, gamma_in, normed, _EPS)
        q = _empty((1, 1, _HQ, _HD), hidden.device)
        k = _empty((1, 1, _HKV, _HD), hidden.device)
        v = _empty((1, 1, _HKV, _HD), hidden.device)
        _K.gemv(w_q, normed, q)
        _K.gemv(w_k, normed, k)
        _K.gemv(w_v, normed, v)
        return q, k, v

    @runtime_func
    def attend(self, q, k_cache, v_cache, cur_pos, mask, w_o):
        # `mask` is the reference's way of saying what `cur_pos` says; the
        # kernel reads the number, so its loop is the live length rather than
        # the whole capacity.
        ctx = _empty((1, _HQ * _HD), q.device)
        part_acc = _empty((_HQ, _ATTN_SPLITS, _HD), q.device, torch.float32)
        part_ms = _empty((_HQ, _ATTN_SPLITS, 2), q.device, torch.float32)
        _K.attn_decode(
            q, k_cache, v_cache, cur_pos, part_acc, part_ms, ctx, _HQ, _HKV, sem._ATT_SCALE
        )
        out = _empty((1, 1, _H), q.device)
        _K.gemv(w_o, ctx, out)
        return out

    @runtime_func
    def full_attention(
        self, hidden, gamma_in, w_q, w_k, w_v, k_cache, v_cache, cur_pos, mask, w_o
    ):
        q, k, v = self.qkv(hidden)
        # `cache_update` at fixed capacity: one position written, the tensor
        # handed straight back.
        _K.cache_write(k_cache, k, cur_pos)
        _K.cache_write(v_cache, v, cur_pos)
        return self.attend(q, k_cache, v_cache, cur_pos, mask), k_cache, v_cache


# ---------------------------------------------------------------------------
# Mixture of experts
# ---------------------------------------------------------------------------


@runtime_module(sem.GraniteRouter)
class Router:
    @runtime_func
    def routing(self, tokens, w_router):
        # An ordinary matvec for the 72 logits, then one warp for the top ten
        # and their softmax. Selecting inside the projection kernel would tie
        # 590 KB of reads to a single block, which was the slowest kernel in
        # the whole step before it was split.
        logits = _empty((_E,), tokens.device, torch.float32)
        _K.gemv_f32(w_router, tokens, logits, 1.0)
        weights = _empty((1, _KTOP), tokens.device)
        indices = _empty((1, _KTOP), tokens.device, torch.int64)
        _K.topk_softmax(logits, weights, indices)
        return weights, indices


@runtime_module(sem.GraniteMoE)
class MoE:
    router = Router

    @runtime_func
    def post_norm(self, hidden, gamma_post):
        out = _empty((1, _H), hidden.device)
        _K.rms_norm(hidden, gamma_post, out, _EPS)
        return out

    @runtime_func
    def routed_experts(self, tokens, weights, indices, w_gate, w_up, w_down):
        # Two kernels for ten experts. The first reads each expert's gate and
        # up rows together; the second walks all ten down projections into one
        # f32 accumulator per output row, so the ten partial results never go
        # back to memory.
        both = _empty((_KTOP, 2 * _I), tokens.device)
        inner = _empty((_KTOP, _I), tokens.device)
        _K.experts_gate_up(w_gate, w_up, tokens, indices, both, inner)
        partial = _empty((_KTOP, _H), tokens.device, torch.float32)
        out = _empty((1, _H), tokens.device)
        _K.experts_down(w_down, inner, indices, weights, partial, out)
        return out

    @runtime_func
    def shared_mlp(self, tokens, w_shared_in, w_shared_out):
        # The fused input projection is one matvec over all 3072 rows -- twice
        # the rows in flight that projecting the two halves separately gives --
        # and the SwiGLU joining them is its own elementwise pass.
        both = _empty((2 * _IS,), tokens.device)
        _K.gemv(w_shared_in, tokens, both)
        inner = _empty((_IS,), tokens.device)
        _K.swiglu(both, inner)
        out = _empty((1, _H), tokens.device)
        _K.gemv(w_shared_out, inner, out)
        return out

    @runtime_func
    def experts(
        self, tokens, weights, indices, w_gate, w_up, w_down, w_shared_in, w_shared_out
    ):
        routed = self.routed_experts(tokens, weights, indices)
        shared = self.shared_mlp(tokens)
        out = _empty((1, 1, _H), tokens.device)
        _K.residual_add(routed, shared, out, 1.0)
        return out


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


def _residual_add(self, a, b):
    out = _empty(a.shape, a.device)
    _K.residual_add(a, b, out, sem._RES_MULT)
    return out


@runtime_module(sem.GraniteMambaDecoderLayer)
class MambaDecoderLayer:
    mixer = MambaMixer
    moe = MoE
    residual_add = runtime_func(_residual_add)


@runtime_module(sem.GraniteAttentionDecoderLayer)
class AttentionDecoderLayer:
    mixer = Attention
    moe = MoE
    residual_add = runtime_func(_residual_add)


_LAYER_TWIN = {
    "linear_attention": MambaDecoderLayer,
    "full_attention": AttentionDecoderLayer,
}


# ---------------------------------------------------------------------------
# The root, and the decode loop it drives
# ---------------------------------------------------------------------------

#: How many partial maxima the sampler reduces over. One per block, so this is
#: also the sampler's grid.
_SAMPLE_PARTS = 256


class _Root:
    """Body of the root twin. The forty layer children are attached below,
    because their classes are chosen by `config.layer_types` and a class body
    cannot name forty attributes it computes."""

    @runtime_func
    def embed(self, table, token_ids):
        out = _empty((1, 1, _H), table.device)
        _K.embed(table, token_ids, out, sem._EMB_MULT)
        return out

    @runtime_func
    def final_rms_norm(self, hidden, gamma_final):
        out = _empty((1, 1, _H), hidden.device)
        _K.rms_norm(hidden, gamma_final, out, _EPS)
        return out

    @runtime_func
    def lm_head(self, hidden, table):
        # f32 out: these logits are what the sampler compares, and a bf16
        # logit has eight mantissa bits to separate a hundred thousand of them.
        out = _empty((1, _V), hidden.device, torch.float32)
        _K.gemv_f32(table, hidden, out, 1.0 / sem._LOGIT_SCALE)
        return out

    # -- orchestration ------------------------------------------------------

    def init_caches(self, device=None):
        """The same per-layer containers the authored Module declares, at the
        same shapes -- allocated once, because the kernels advance them in
        place and `generate` needs their addresses to stay put across a graph
        replay."""
        device = torch.accelerator.current_accelerator() if device is None else device
        entries = []
        for kind in sem.config.layer_types:
            if kind == "linear_attention":
                entries.append((
                    torch.zeros(1, _CONVD, _WIN, dtype=_BF, device=device),
                    torch.zeros(1, _NH, _PD, _NS, dtype=torch.float32, device=device),
                ))
            else:
                entries.append((
                    torch.zeros(1, _CAP, _HKV, _HD, dtype=_BF, device=device),
                    torch.zeros(1, _CAP, _HKV, _HD, dtype=_BF, device=device),
                ))
        return tuple(entries)

    def append_cache(self, caches, fresh):
        """Identity: every kernel that produced *fresh* wrote it into the
        tensor it was handed, so the caches already hold this step's state."""
        return caches

    def reset_caches(self, caches):
        """Zero every layer's state, which is what a fresh sequence starts
        from -- and what a warm-up pass has to be undone back to."""
        for entry in caches:
            for tensor in entry:
                tensor.zero_()

    # -- the decode loop ----------------------------------------------------

    def generate(
        self, prompt_ids, *, max_new, greedy=False, seed=0, temperature=1.0, eos=(),
        device=None, capture=True,
    ):
        """Decode `max_new` tokens after *prompt_ids*, and time the decode.

        Everything a step reads or writes lives at a fixed address on the
        device -- the token, the position, the caches, the sampler's output --
        so the step captures into a CUDA graph once and replays per token. The
        host learns the token only to test it against *eos*.
        """
        device = torch.accelerator.current_accelerator() if device is None else device
        prompt = list(prompt_ids)
        if not prompt:
            raise ValueError("generate needs a prompt of at least one token")
        if len(prompt) + max_new > _CAP:
            raise ValueError(
                f"prompt {len(prompt)} + {max_new} new exceeds the {_CAP}-position "
                f"capacity model.py declares; raise GRANITE_MAX_CTX"
            )

        caches = self.init_caches(device=device)
        token = torch.zeros(1, dtype=torch.int64, device=device)
        cur_pos = torch.zeros(1, dtype=torch.int32, device=device)
        # Declared because the reference declares it; `cur_pos` is what the
        # kernel reads.
        mask = torch.zeros(1, 1, _CAP, dtype=_BF, device=device)
        sampled = torch.zeros(len(prompt) + max_new + 1, dtype=torch.int32, device=device)
        part_val = torch.zeros(_SAMPLE_PARTS, dtype=torch.float32, device=device)
        part_idx = torch.zeros(_SAMPLE_PARTS, dtype=torch.int32, device=device)
        layer_args = tuple(
            () if kind == "linear_attention" else (cur_pos, mask)
            for kind in sem.config.layer_types
        )

        def step():
            logits, fresh = self.forward(token, layer_args, caches)
            self.append_cache(caches, fresh)
            # The sampler writes the next input token, records this step's
            # choice, and advances the position -- all on the device, so the
            # next replay needs nothing from the host.
            _K.sample(
                logits, part_val, part_idx, token, sampled, cur_pos,
                seed, greedy, temperature, True,
            )

        graph = None
        if capture:
            # Warm up on a side stream so the allocator has its pool, then undo
            # what the warm-up did to the state before recording.
            warm = torch.cuda.Stream(device=device)
            warm.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warm):
                for _ in range(2):
                    step()
            torch.cuda.current_stream(device).wait_stream(warm)
            torch.cuda.synchronize(device)
            self.reset_caches(caches)
            cur_pos.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            self.reset_caches(caches)
            cur_pos.zero_()

        def run_step():
            if graph is not None:
                graph.replay()
            else:
                step()

        # Prefill: every prompt token but the last. Each step's sampler leaves
        # its own prediction in `token`, which the next prompt token overwrites.
        for index in range(len(prompt) - 1):
            token.fill_(prompt[index])
            run_step()
        token.fill_(prompt[-1])
        torch.cuda.synchronize(device)

        # Decode: `max_new` steps, `max_new` tokens, nothing else in the timing.
        host = torch.zeros(1, dtype=torch.int64, device="cpu", pin_memory=True)
        produced = []
        started = perf_counter()
        for _ in range(max_new):
            run_step()
            host.copy_(token, non_blocking=True)
            torch.cuda.synchronize(device)
            produced.append(int(host.item()))
            if produced[-1] in eos:
                break
        elapsed = perf_counter() - started
        return produced, elapsed


def _root_namespace():
    """`_Root`'s own members plus one layer child per published layer type."""
    namespace = {
        name: value for name, value in vars(_Root).items() if not name.startswith("__")
    }
    namespace.update({
        f"layer{index}": _LAYER_TWIN[kind]
        for index, kind in enumerate(sem.config.layer_types)
    })
    return namespace


#: The runtime twin of the whole model. Built with `type` rather than a class
#: statement because its children are named from `config.layer_types`.
Granite4_0_H_Small = runtime_module(sem.Granite4_0_H_Small)(
    type("Granite4_0_H_Small", (), _root_namespace())
)


__all__ = [
    "Attention",
    "AttentionDecoderLayer",
    "Granite4_0_H_Small",
    "MambaDecoderLayer",
    "MambaMixer",
    "MoE",
    "Router",
]
