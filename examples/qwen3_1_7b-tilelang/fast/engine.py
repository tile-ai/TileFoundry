"""One captured decode step, replayed.

The authored reference hands each step's key and value back for the caller to
``torch.cat`` on. That is the right contract for a reference -- it keeps every
shape expressed in ``ctx_len`` alone -- but it means the cache buffer moves
every step, and a graph records addresses, so nothing built that way can be
replayed. This engine takes the other form the migrate page names: a cache of
fixed capacity whose write window advances, with the position in a one-element
device tensor.

Everything a step needs then has a fixed address, so the whole step -- 284
kernels, embedding through the greedy pick -- is captured once and replayed. The
chosen token is written back into the input slot by the last kernel, and while
the prompt still has a token left that kernel feeds that one instead, so the
same capture walks the prompt and continues past it with no host round trip
anywhere in the loop.

Weights are repacked once at load: ``q|k|v`` become one matrix and ``gate|up``
another, because a decode GEMV is bandwidth-bound and its cost is the block
count it can fill, not the arithmetic. Two fused reads beat five thin ones.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

import kernels as K

#: Tile shapes per projection, chosen by measuring each under a graph replay:
#: ``(BN, BK, SK, threads)``. ``SK`` is what keeps an H200's 132 SMs busy on the
#: narrow projections -- ``N/BN`` blocks alone leaves most of them idle.
TILES = {
    "qkv": (256, 64, 8, 256),
    "o": (128, 128, 8, 128),
    "gate_up": (256, 64, 2, 256),
    "down": (128, 128, 8, 128),
    "head": (128, 128, 128),
}
#: Context positions one attention block owns.
SPLIT = 256


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Generated:
    """A continuation and the timing of the part that produced it."""

    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


class Engine:
    """A loaded Qwen3-1.7B and one captured decode step over it."""

    def __init__(self, ckpt: str | Path, ref_dir: str | Path, *, device: str = "cuda:0",
                 max_new: int = 2048, prompt_room: int = 512):
        from tilefoundry.runtime import SafetensorsResource

        ref_dir = Path(ref_dir)
        self.ref = _load_module(ref_dir / "model.py", "ref_model")
        alias_mod = _load_module(ref_dir / "hf_alias.py", "ref_alias")
        cfg = self.ref.config
        self.cfg = cfg
        self.device = device
        self.dt = torch.bfloat16

        self.H = cfg.hidden_size
        self.HQ = cfg.num_attention_heads
        self.HKV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.I = cfg.intermediate_size
        self.V = cfg.vocab_size
        self.L = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.scale = self.D ** -0.5
        self.qkv_n = self.HQ * self.D + 2 * self.HKV * self.D

        self.nsteps = max_new + prompt_room
        self.cap = ((self.nsteps + SPLIT - 1) // SPLIT) * SPLIT
        self.ns = self.cap // SPLIT

        loaded = self.ref.Qwen3_1_7B.load(
            SafetensorsResource(str(ckpt), device=device, alias=alias_mod.hf_alias(cfg))
        )
        self._pack(loaded)
        del loaded
        torch.cuda.empty_cache()
        self._buffers()
        self._kernels()
        self._capture()

    # ---------------------------------------------------------------- weights
    def _pack(self, loaded):
        """Fuse the projections a single GEMV can serve, and keep the rest as is."""
        self.w_embed = loaded.constants["w_embed"]
        self.w_head = loaded.constants["w_head"]
        self.gamma_final = loaded.constants["gamma_final"]
        self.layers = []
        for i in range(self.L):
            c = getattr(loaded, f"layer{i}").constants
            self.layers.append({
                "gamma_in": c["gamma_in"],
                "gamma_post": c["gamma_post"],
                "gamma_q": c["gamma_q"],
                "gamma_k": c["gamma_k"],
                # [q | k | v] over the output axis: the rope kernel splits it back
                "w_qkv": torch.cat([c["w_q"][0], c["w_k"][0], c["w_v"][0]], dim=1)
                          .contiguous(),
                "w_o": c["w_o"][0].contiguous(),
                # [gate | up]: silu_mul consumes both halves of one partial
                "w_gu": torch.cat([c["w_gate"][0], c["w_up"][0]], dim=1).contiguous(),
                "w_down": c["w_down"][0].contiguous(),
            })
        cos, sin = self.ref._generation_rope(self.device)
        self.cos = cos[: self.cap].contiguous()
        self.sin = sin[: self.cap].contiguous()

    # ---------------------------------------------------------------- buffers
    def _buffers(self):
        dev, dt, f32 = self.device, self.dt, torch.float32

        def z(*shape, dtype=dt):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        self.hid = z(self.H)
        self.xn = z(self.H)
        self.h1 = z(self.H)
        self.xn1 = z(self.H)
        self.q = z(self.HQ * self.D)
        self.qkv_part = z(TILES["qkv"][2], self.qkv_n, dtype=f32)
        self.o_part = z(TILES["o"][2], self.H, dtype=f32)
        self.gu_part = z(TILES["gate_up"][2], 2 * self.I, dtype=f32)
        self.d_part = z(TILES["down"][2], self.H, dtype=f32)
        self.op = z(self.ns, self.HQ, self.D, dtype=f32)
        self.mp = z(self.ns, self.HQ, dtype=f32)
        self.lp = z(self.ns, self.HQ, dtype=f32)
        self.logits = z(self.V, dtype=f32)
        # A zeroed cache is what makes masking enough: a position past the end
        # contributes exp(-inf) * 0, not exp(-inf) * whatever was there.
        self.kc = z(self.L, self.cap, self.HKV * self.D)
        self.vc = z(self.L, self.cap, self.HKV * self.D)
        self.pos = z(1, dtype=torch.int32)
        self.ids = z(1, dtype=torch.int64)
        self.inp = z(self.nsteps, dtype=torch.int32)
        self.sam = z(self.nsteps, dtype=torch.int32)
        self.plen = z(1, dtype=torch.int32)

    # ---------------------------------------------------------------- kernels
    def _kernels(self):
        H, HQ, HKV, D, I, V = self.H, self.HQ, self.HKV, self.D, self.I, self.V
        self.k_embed = K.embed(V, H)
        self.k_norm = K.rms_norm(H, self.eps)
        self.k_qkv = K.gemv(H, self.qkv_n, *TILES["qkv"])
        self.k_rope = K.qk_rope_cache(
            HQ, HKV, D, self.cap, self.cap, TILES["qkv"][2], self.eps
        )
        self.k_attn = K.attn_partial(HQ, HKV, D, self.cap, SPLIT, self.scale)
        # o_proj merges the attention splits itself; down_proj activates its own
        # slice of the gate/up partial. Two fewer launches per layer.
        self.k_o = K.gemv_attn_combine(HQ, D, H, *TILES["o"][:3], self.ns,
                                       TILES["o"][3])
        self.k_rn_post = K.resid_rms_norm(H, TILES["o"][2], self.eps)
        self.k_gu = K.gemv(H, 2 * I, *TILES["gate_up"])
        self.k_down = K.gemv_silu(I, H, *TILES["down"][:3],
                                  TILES["gate_up"][2], TILES["down"][3])
        self.k_rn_in = K.resid_rms_norm(H, TILES["down"][2], self.eps)
        self.k_head, self.nb = K.lm_head(H, V, *TILES["head"])
        self.bv = torch.zeros(self.nb, device=self.device, dtype=torch.float32)
        self.bi = torch.zeros(self.nb, device=self.device, dtype=torch.int32)
        self.k_sample = K.sample_step(self.nb, self.nsteps)

    def _step(self, record=None):
        """One decode step, as the sequence of launches the graph records.

        *record*, when given, collects the hidden state entering the stack and
        leaving each layer -- the same series ``output_hidden_states`` returns,
        so a disagreement can be pinned to a layer instead of to the model.
        """
        self.k_embed(self.w_embed, self.ids, self.hid)
        self.k_norm(self.hid, self.layers[0]["gamma_in"], self.xn)
        if record is not None:
            record.append(self.hid.clone())
        for i, w in enumerate(self.layers):
            kc, vc = self.kc[i], self.vc[i]
            self.k_qkv(self.xn, w["w_qkv"], self.qkv_part)
            self.k_rope(self.qkv_part, w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                        self.pos, self.pos, kc, vc, self.q)
            self.k_attn(self.q, kc, vc, self.pos, self.op, self.mp, self.lp)
            self.k_o(self.op, self.mp, self.lp, w["w_o"], self.o_part)
            self.k_rn_post(self.hid, self.o_part, w["gamma_post"], self.h1, self.xn1)
            self.k_gu(self.xn1, w["w_gu"], self.gu_part)
            self.k_down(self.gu_part, w["w_down"], self.d_part)
            # The next layer's input norm, or the norm that closes the stack --
            # so the residual add never needs a kernel to itself.
            nxt = (self.layers[i + 1]["gamma_in"] if i + 1 < self.L else self.gamma_final)
            self.k_rn_in(self.h1, self.d_part, nxt, self.hid, self.xn)
            if record is not None:
                record.append(self.hid.clone())
        self.k_head(self.xn, self.w_head, self.logits, self.bv, self.bi)
        self.k_sample(self.bv, self.bi, self.inp, self.plen, self.ids, self.pos, self.sam)

    def _capture(self):
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self._reset()

    def _reset(self):
        self.pos.zero_()
        self.kc.zero_()
        self.vc.zero_()
        self.sam.zero_()

    # ------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new: int) -> Generated:
        """Walk *prompt_ids*, then continue for *max_new* tokens.

        Timing covers exactly the steps that produce the continuation: the step
        at ``pos = len(prompt) - 1`` is the first one whose pick is kept, so the
        steps before it are prefill and are not counted.
        """
        pl = len(prompt_ids)
        if pl < 1:
            raise ValueError("decode needs a prompt of at least one token")
        if pl + max_new > self.nsteps:
            raise ValueError(
                f"prompt {pl} + {max_new} new exceeds this engine's {self.nsteps} steps"
            )
        self._reset()
        self.inp.zero_()
        self.inp[:pl] = torch.tensor(prompt_ids, device=self.device, dtype=torch.int32)
        self.plen.fill_(pl)
        self.ids.fill_(prompt_ids[0])

        torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        for _ in range(pl - 1):                     # prefill: picks discarded
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        t1 = perf_counter()
        for _ in range(max_new):                    # the continuation itself
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        t2 = perf_counter()

        out = self.sam[pl - 1: pl - 1 + max_new].tolist()
        return Generated(out, t2 - t1, t1 - t0, pl)

    def trace(self, prompt_ids: list[int]) -> list[torch.Tensor]:
        """Per-layer hidden states at the last prompt position."""
        self._reset()
        self.inp.zero_()
        self.inp[: len(prompt_ids)] = torch.tensor(
            prompt_ids, device=self.device, dtype=torch.int32
        )
        self.plen.fill_(len(prompt_ids))
        self.ids.fill_(prompt_ids[0])
        for _ in range(len(prompt_ids) - 1):
            self.graph.replay()
        rec: list[torch.Tensor] = []
        self._step(record=rec)
        torch.cuda.synchronize(self.device)
        return rec

    def logits_for(self, prompt_ids: list[int]) -> torch.Tensor:
        """The logits after consuming every token of *prompt_ids* -- for checking."""
        self._reset()
        self.inp.zero_()
        self.inp[: len(prompt_ids)] = torch.tensor(
            prompt_ids, device=self.device, dtype=torch.int32
        )
        self.plen.fill_(len(prompt_ids))
        self.ids.fill_(prompt_ids[0])
        for _ in range(len(prompt_ids)):
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        return self.logits.clone()


def default_ref_dir() -> Path:
    """The pristine copy of the shipped source this engine is measured against."""
    here = Path(__file__).resolve().parent
    return here.parent / "ref_src"
