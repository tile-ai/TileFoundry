"""`model.py`'s high-performance implementation: tilelang kernels behind
`@runtime_module` twins, plus the driver that turns one step into many.

Two layers, and they answer different questions.

**The twins** (`Qwen3_5LinearAttention` ... `Qwen3_5Decoder`) are runtime
counterparts of `model.py`'s Modules, one `@runtime_func` per authored `@func`,
same names, same order, same shapes. `@runtime_module` validates that
correspondence at decoration time and `tilefoundry check` measures it, so these
are the thing that is *judged*. They keep the authored contract exactly: state
arrives as a parameter and leaves as a result, nothing is mutated in place, and
the prior cache is however long the caller says it is.

**The driver** (`Session`) is the thing that is *fast*. Under the authored
contract a decode step is ~500 kernel calls, and an eager tilelang call costs
20.5 us of pure Python -- 10 ms a token, of which 95% is Python. So the driver
captures the step into a CUDA graph, where a launch costs 0.91 us. A CUDA graph
replays fixed addresses and fixed shapes, which the authored contract's growing
`k_cache: (1, ctx_len, 2, 256)` cannot provide; so the graphed path uses a
fixed-capacity cache and a device-side position instead, and says so.

`Session` has both, selected by `--graph` / `--no-graph`, and `run.py --check`
asserts they produce the same tokens. That is the point of keeping both: the
eager path is the contract, the graphed path is the speed, and neither is taken
on trust.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import model as sem  # noqa: E402
import weights as wt  # noqa: E402
from config import CKPT, REAL  # noqa: E402
from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

# ---------------------------------------------------------------------------
# Which implementation each function uses.
# ---------------------------------------------------------------------------
#
# A decode step is ~500 kernel calls. When the logits come out wrong, "one of
# these 500" is not a diagnosis, so every function is independently switchable:
#
#   TF_IMPL=torch                     everything in plain torch
#   TF_IMPL=linear_attention:torch    that one function in torch, the rest tilelang
#
# The torch bodies are in `kernels/torch_ref.py` and are a transcription of the
# authored `@func` bodies. Bisecting a wrong output is then a handful of runs
# rather than a reading of every kernel.

_IMPL_SPEC = os.environ.get("TF_IMPL", "").strip()
_ALL_TORCH = _IMPL_SPEC == "torch"
_PER_FN = {}
if _IMPL_SPEC and not _ALL_TORCH:
    for entry in _IMPL_SPEC.split(","):
        name, _, which = entry.partition(":")
        _PER_FN[name.strip()] = (which or "torch").strip()


def _impl(name, fast, slow):
    """The callable for *name*: the tilelang one unless asked for otherwise."""
    if _ALL_TORCH or _PER_FN.get(name) == "torch":
        return slow
    return fast


from kernels import basic as _basic, torch_ref as _tr  # noqa: E402


def _optional(name):
    """A kernel module, or a stand-in that explains itself when called.

    A tilelang module that is absent or fails to import must not stop
    `TF_IMPL=torch` from running -- that path is how a kernel gets debugged, so
    it cannot depend on the kernel importing. Deferring the complaint to the
    call also means the failure names the one function that was reached.
    """
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(f"kernels.{name}")
    except Exception as error:  # pragma: no cover - bring-up path
        class _Absent:
            def __getattr__(self, fn):
                def _refuse(*_a, **_k):
                    raise RuntimeError(
                        f"kernels/{name}.py did not import, so {fn!r} has no "
                        f"tilelang implementation ({type(error).__name__}: "
                        f"{error}). Run with TF_IMPL={fn}:torch, or TF_IMPL=torch."
                    )
                return _refuse

        return _Absent()


_gdn = _optional("gdn")
_attn = _optional("attn")
_moe = _optional("moe")

_CFG = REAL


# ---------------------------------------------------------------------------
# The twins.
# ---------------------------------------------------------------------------


@runtime_module(sem.Qwen3_5LinearAttention)
class Qwen3_5LinearAttention:
    @runtime_func
    def conv_step(self, conv_state, entry, conv_w):
        return _impl("conv_step", _gdn.conv_step, _tr.conv_step)(
            conv_state, entry, conv_w
        )

    @runtime_func
    def l2_normalise(self, x):
        return _impl("l2_normalise", _gdn.l2_normalise, _tr.l2_normalise)(x)

    @runtime_func
    def delta_step(self, recurrent_state, q, k, v, g, beta):
        return _impl("delta_step", _gdn.delta_step, _tr.delta_step)(
            recurrent_state, q, k, v, g, beta
        )

    @runtime_func
    def linear_attention(
        self, hidden, gamma_in, w_in_qkv, w_in_z, w_in_b, w_in_a, conv_w, a_log,
        dt_bias, conv_state, recurrent_state, gamma_gdn, w_out,
    ):
        return _impl(
            "linear_attention", _gdn.linear_attention, _tr.linear_attention
        )(
            hidden, gamma_in, w_in_qkv, w_in_z, w_in_b, w_in_a, conv_w, a_log,
            dt_bias, conv_state, recurrent_state, gamma_gdn, w_out,
        )


@runtime_module(sem.Qwen3_5FullAttention)
class Qwen3_5FullAttention:
    @runtime_func
    def partial_rope(self, x, cos_cache, sin_cache, pos_ids):
        return _impl("partial_rope", _attn.partial_rope, _tr.partial_rope)(
            x, cos_cache, sin_cache, pos_ids
        )

    @runtime_func
    def partial_rope_kv(self, x, cos_cache, sin_cache, pos_ids):
        return _impl(
            "partial_rope_kv", _attn.partial_rope_kv, _tr.partial_rope_kv
        )(x, cos_cache, sin_cache, pos_ids)

    @runtime_func
    def full_attention(
        self, hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache,
        sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
    ):
        return _impl(
            "full_attention", _attn.full_attention, _tr.full_attention
        )(
            hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache,
            sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
        )


@runtime_module(sem.Qwen3_5Router)
class Qwen3_5Router:
    @runtime_func
    def routing(self, tokens, w_router):
        return _impl("routing", _moe.routing, _tr.routing)(tokens, w_router)


@runtime_module(sem.Qwen3_5MoE)
class Qwen3_5MoE:
    router = Qwen3_5Router

    @runtime_func
    def post_norm(self, hidden, gamma_post):
        return _impl("post_norm", _moe.post_norm, _tr.post_norm)(
            hidden, gamma_post
        )

    @runtime_func
    def routed_experts(self, tokens, weights, indices, w_gate, w_up, w_down):
        return _impl(
            "routed_experts", _moe.routed_experts, _tr.routed_experts
        )(tokens, weights, indices, w_gate, w_up, w_down)

    @runtime_func
    def shared_expert(
        self, tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
    ):
        return _impl("shared_expert", _moe.shared_expert, _tr.shared_expert)(
            tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
        )

    @runtime_func
    def experts(
        self, tokens, weights, indices, w_gate, w_up, w_down, w_shared_gate,
        w_shared_up, w_shared_down, w_shared_scale,
    ):
        return _impl("experts", _moe.experts, _tr.experts)(
            tokens, weights, indices, w_gate, w_up, w_down, w_shared_gate,
            w_shared_up, w_shared_down, w_shared_scale,
        )


def _residual_add(self, a, b):
    """The layer's residual. One function, two layer kinds, one body."""
    return _impl("residual_add", _tl_add, _tr.residual_add)(a, b)


def _tl_add(a, b):
    out = torch.empty_like(a)
    _basic.add(a.numel())(a.reshape(-1), b.reshape(-1), out.reshape(-1))
    return out


Qwen3_5FullAttnLayer = runtime_module(sem.Qwen3_5FullAttnLayer)(
    type(
        "Qwen3_5FullAttnLayer",
        (),
        {"mixer": Qwen3_5FullAttention, "moe": Qwen3_5MoE,
         "residual_add": runtime_func(_residual_add)},
    )
)

Qwen3_5LinearAttnLayer = runtime_module(sem.Qwen3_5LinearAttnLayer)(
    type(
        "Qwen3_5LinearAttnLayer",
        (),
        {"mixer": Qwen3_5LinearAttention, "moe": Qwen3_5MoE,
         "residual_add": runtime_func(_residual_add)},
    )
)

_LAYER_TWIN = {
    "full_attention": Qwen3_5FullAttnLayer,
    "linear_attention": Qwen3_5LinearAttnLayer,
}


def _embed(self, table, token_ids):
    def _fast(table, token_ids):
        out = torch.empty(
            (1, 1, table.shape[1]), dtype=torch.float32, device=table.device
        )
        _basic.embed_row(table.shape[0], table.shape[1])(
            table, token_ids, out.reshape(-1)
        )
        return out

    return _impl("embed", _fast, _tr.embed)(table, token_ids)


def _final_rms_norm(self, hidden, gamma_final):
    def _fast(hidden, gamma_final):
        out = torch.empty_like(hidden)
        _basic.rms_norm(hidden.shape[-1], _CFG.rms_eps)(
            hidden.reshape(-1), gamma_final, out.reshape(-1)
        )
        return out

    return _impl("final_rms_norm", _fast, _tr.final_rms_norm)(hidden, gamma_final)


def _lm_head(self, hidden, w_head):
    def _fast(hidden, w_head):
        out = torch.empty(
            (1, w_head.shape[1]), dtype=torch.float32, device=hidden.device
        )
        _basic.gemv(w_head.shape[0], w_head.shape[1])(
            hidden.reshape(-1), w_head, out.reshape(-1)
        )
        return out

    return _impl("lm_head", _fast, _tr.lm_head)(hidden, w_head)


def _decoder_twin(cfg=REAL, sem_decoder=None):
    """The decoder twin of *sem_decoder*, at *cfg*'s layer-type cycle.

    Built with `type()` because `@runtime_module` reads the child classes out of
    `vars(cls)` and requires one class attribute per authored child -- 40 of
    them, named `layer0` .. `layer39` by the authored side, and *exactly* those
    (missing or extra is rejected at decoration time). A class body cannot write
    that from a tuple.

    *sem_decoder* has to be the Module at the same layer count, which is why a
    truncated run builds its own semantic tree through `model.build(cfg)` rather
    than reusing the published one.
    """
    sem_decoder = sem.Qwen3_5Decoder if sem_decoder is None else sem_decoder
    namespace = {
        f"layer{index}": _LAYER_TWIN[kind]
        for index, kind in enumerate(cfg.layer_types)
    }
    namespace.update(
        embed=runtime_func(_embed),
        final_rms_norm=runtime_func(_final_rms_norm),
        lm_head=runtime_func(_lm_head),
    )
    return runtime_module(sem_decoder)(type("Qwen3_5Decoder", (), namespace))


Qwen3_5Decoder = _decoder_twin(REAL)


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


class Session:
    """One loaded decoder, driven a token at a time.

    Owns the things a step needs that are not weights: the rope caches, the
    attention scale, the per-layer state, and -- on the graphed path -- the
    fixed-capacity cache and the captured graph.
    """

    def __init__(
        self,
        cfg=REAL,
        *,
        # Where the checkpoint is has to arrive from the caller; `CKPT` is only
        # non-None when the environment named it.
        ckpt=CKPT,
        device="cuda",
        dtype=torch.bfloat16,
        capacity: int = 1024,
        verbose=False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.capacity = min(capacity, cfg.max_ctx)
        # The semantic tree and the twin must be the same shape: `@runtime_module`
        # requires the child-name sets to be *equal*, so a truncated run needs its
        # own authored tree rather than the published one with layers ignored.
        self.sem = sem.Qwen3_5Decoder if cfg is REAL else sem.build(cfg)["Qwen3_5Decoder"]
        twin_cls = Qwen3_5Decoder if cfg is REAL else _decoder_twin(cfg, self.sem)
        self.twin = twin_cls()

        t0 = time.perf_counter()
        resource, tally = wt.decoder_resource(
            self.sem, ckpt=ckpt, cfg=cfg, device=device, dtype=dtype, verbose=verbose
        )
        self.twin.load(resource)
        self.load_seconds = time.perf_counter() - t0
        self.loaded_bytes = tally["bytes"]

        self.cos, self.sin = wt.rope_caches(cfg, device=device)
        self.scale = torch.full(
            (1, 1, 1, 1), cfg.attn_scale, dtype=torch.float32, device=device
        )
        self.reset()

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Back to position zero: empty caches, zero conv window and state."""
        self.caches = list(self.twin.init_caches(device=self.device))
        self.pos = 0

    def _layer_args(self, pos: int):
        """One tuple per layer of what its mixer takes besides hidden and state.

        A linear-attention mixer takes nothing else; a full-attention one takes
        the rope caches, this token's position, and the attention scale. Which
        slot the state is spliced into is the authored side's business
        (`_with_cache`), not this function's.
        """
        pos_ids = torch.tensor([pos], dtype=torch.int32, device=self.device)
        return tuple(
            () if kind == "linear_attention"
            else (self.cos, self.sin, pos_ids, self.scale)
            for kind in self.cfg.layer_types
        )

    # -- the eager path: the authored contract, exactly -------------------

    def step(self, token_id: int):
        """One decode step through the authored orchestration. Returns logits.

        `forward` here is `model.py`'s own `Qwen3_5Decoder.forward`, reused
        verbatim -- runtime §1.1 requires that, and it is what makes this the
        *same* step the reference takes rather than a re-implementation of it.
        """
        ids = torch.tensor([token_id], dtype=torch.int64, device=self.device)
        logits, fresh = self.twin.forward(ids, self._layer_args(self.pos), tuple(self.caches))
        self.caches = list(self.twin.append_cache(tuple(self.caches), fresh))
        self.pos += 1
        return logits

    # -- the graphed path -------------------------------------------------
    #
    # See `graph.py`. Kept out of this class so the contract-faithful part above
    # stays readable, and so a twin can be checked without any of it existing.

    def graphed(self):
        from graph import GraphedStep  # noqa: PLC0415

        return GraphedStep(self)


__all__ = [
    "Qwen3_5Decoder",
    "Qwen3_5FullAttention",
    "Qwen3_5FullAttnLayer",
    "Qwen3_5LinearAttention",
    "Qwen3_5LinearAttnLayer",
    "Qwen3_5MoE",
    "Qwen3_5Router",
    "Session",
]
