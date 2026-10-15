"""The tilelang prefill engine: static buffers for one prompt length, the kernel
sequence of each layer kind, and one CUDA graph per length.

The runtime twin (runtime/twin.py) calls into this module; nothing here computes a
value with torch -- torch only allocates, views, and copies the three inputs into the
static buffers the captured graph reads.

Per GDN layer (h_in -> h_out), kernels in order:
  rmsnorm(h_in, 1+gamma_in) -> x
  gemm  x @ [w_qkv | w_z | w_b | w_a]^T -> proj (T, 16480)          (packed once)
  gdn.prep  proj -> q, k, v, g, beta, conv window
  gdn.core  -> y (T, 6144) = gated RMSNorm(delta rule), final state
  gemm residual  hmid = h_in + y @ w_out^T
  rmsnorm(hmid, 1+gamma_post) -> x
  gemm swiglu2   act = silu(x @ w_gate^T) * (x @ w_up^T)
  gemm residual  h_out = hmid + act @ w_down^T
Per attention layer the mixer part is
  gemm x @ [w_q | w_k | w_v]^T -> qkv (T, 14336);  attn.prep -> q, K/V cache rows;
  attn.core -> y (T, 6144) = attention * sigmoid(gate);  gemm residual with w_o.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from runtime import plan
from runtime.kernels import small
from runtime.kernels.gemm_ws import gemm_ws_kernel
from runtime.kernels.gemm import BA_N, ba_bn, ba_reduce_kernel, ba_split, ba_splitk_kernel, gemm_kernel, splitk_reduce_kernel

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())["text_config"]
H = CFG["hidden_size"]
I = CFG["intermediate_size"]
V = CFG["vocab_size"]
KINDS = tuple(CFG["layer_types"])
HQ, HKV, D = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
HK, HV, DK, DV = (CFG["linear_num_key_heads"], CFG["linear_num_value_heads"],
                  CFG["linear_key_head_dim"], CFG["linear_value_head_dim"])
CONV = 2 * HK * DK + HV * DV
ROT = int(D * CFG["rope_parameters"]["partial_rotary_factor"])
WINDOW = CFG["linear_conv_kernel_dim"] - 1
GDN_IN = CONV + HV * DV + 2 * HV          # qkv | z | b | a = 16480
GDN_IN_PAD = 16640                         # packed rows padded with zeros to 65 x 256 tiles
ATT_IN = HQ * D * 2 + 2 * HKV * D          # q(+gate) | k | v = 14336
GDN_SLOT = {i: n for n, i in enumerate(i for i, k in enumerate(KINDS) if k == "linear_attention")}
ATT_SLOT = {i: n for n, i in enumerate(i for i, k in enumerate(KINDS) if k == "full_attention")}


class Engine:
    """Everything one prompt length T needs; built once, reused by every call."""

    def __init__(self, T: int, device: torch.device):
        self.T = T
        self.device = device
        e = dict(device=device, dtype=torch.bfloat16)
        self.ids = torch.zeros(T, device=device, dtype=torch.int64)
        self.cos = torch.zeros(T, ROT, **e)
        self.sin = torch.zeros_like(self.cos)
        self.h = [torch.zeros(T, H, **e) for _ in range(3)]  # ping, pong, standalone output
        self.hmid = torch.zeros(T, H, **e)
        self.x = torch.zeros(T, H, **e)
        self.proj = torch.zeros(T, GDN_IN_PAD, **e)
        self.qkv = torch.zeros(T, ATT_IN, **e)
        self.q = torch.zeros(T, HK, DK, **e)
        self.k = torch.zeros(T, HK, DK, **e)
        self.v = torch.zeros(T, HV, DV, **e)
        self.g = torch.zeros(T, HV, device=device, dtype=torch.float32)
        self.beta = torch.zeros(T, HV, **e)
        self.qa = torch.zeros(T, HQ, D, **e)
        self.y = torch.zeros(T, HV * DV, **e)  # == HQ * D == 6144
        self.act = torch.zeros(T, I, **e)
        self.kc = torch.zeros(len(ATT_SLOT), T, HKV, D, **e)
        self.vc = torch.zeros_like(self.kc)
        self.conv = torch.zeros(len(GDN_SLOT), CONV, WINDOW, **e)
        self.rec = torch.zeros(len(GDN_SLOT), HV, DK, DV, device=device, dtype=torch.float32)
        self.xlast = torch.zeros(1, H, **e)
        self.logits = torch.zeros(1, V, **e)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.stream: torch.cuda.Stream | None = None
        self._build_kernels()

    # ----------------------------------------------------------------- kernels
    def _build_kernels(self):
        T = self.T
        p = plan.for_length(T)
        self.plan = p
        self.k_norm = small.rmsnorm_kernel(T, H, p.norm_rows)
        self.k_norm1 = small.rmsnorm_kernel(1, H, 1)
        self.k_embed = small.embed_kernel(T, V, H)
        self.k_head = small.gemv_kernel(V, H, p.head_rows, p.head_threads)
        def lin(name, N, K, epi):
            ws = plan.ws_choice(T, name)
            return gemm_ws_kernel(T, N, K, *ws) if ws else gemm_kernel(T, N, K, p.gemm[name], epi)

        self.k_gdn_in = lin("gdn_in", GDN_IN_PAD, H, "store")
        # b|a recomputed in the reference library's split-K order (gemm.py, BA_N)
        self.k_ba = ba_splitk_kernel(T, H, ba_split(T), BN=ba_bn(T))
        self.k_ba_red = ba_reduce_kernel(T, ba_split(T), GDN_IN_PAD, CONV + HV * DV)
        self.ba_part = torch.empty(ba_split(T), T, BA_N, device=self.device, dtype=torch.float32)
        self.k_att_in = lin("att_in", ATT_IN, H, "store")
        self.k_gu = lin("gate_up", I, H, "swiglu2")
        self.res = {}
        for name, K in (("o_proj", HV * DV), ("down", I)):
            cfg = p.gemm[name]
            if plan.ws_choice(T, name):
                self.res[name] = (lin(name, H, K, "residual"), None, None)
            elif cfg.split_k > 1:
                self.res[name] = (gemm_kernel(T, H, K, cfg, "splitk"),
                                  splitk_reduce_kernel(T, H, cfg.split_k, True),
                                  torch.empty(cfg.split_k, T, H, device=self.device, dtype=torch.float32))
            else:
                self.res[name] = (gemm_kernel(T, H, K, cfg, "residual"), None, None)
        from runtime.kernels.attention import AttentionKernels
        from runtime.kernels.gdn import GdnKernels
        self.gdn = GdnKernels(T, proj_stride=GDN_IN_PAD)
        self.att = AttentionKernels(T)

    def _residual_gemm(self, name, a, w, r, c):
        k, red, part = self.res[name]
        if red is None:
            k(a, w, r, c)
        else:
            k(a, w, part)
            red(part, r, c)

    # ----------------------------------------------------------------- weights
    def pack(self, key: str, parts: tuple[torch.Tensor, ...], rows: int | None = None) -> torch.Tensor:
        """Concatenate a layer's input projections once (rows = output features),
        zero-padded to `rows` rows when given."""
        sig = (key, tuple(t.data_ptr() for t in parts))
        hit = _PACKED.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        if rows is not None:
            pad = rows - sum(t.shape[0] for t in parts)
            parts = (*parts, parts[0].new_zeros(pad, parts[0].shape[1]))
        w = torch.cat(parts, dim=0).contiguous()
        _PACKED[key] = (sig, w)
        return w

    # ----------------------------------------------------------------- layers
    def out_hidden(self, h_in: torch.Tensor) -> torch.Tensor:
        if h_in.data_ptr() == self.h[0].data_ptr():
            return self.h[1]
        if h_in.data_ptr() == self.h[1].data_ptr():
            return self.h[0]
        return self.h[2]

    def mlp(self, hmid, h_out, gamma_post, w_gate, w_up, w_down):
        self.k_norm(hmid, gamma_post, self.x)
        self.k_gu(self.x, w_gate, w_up, self.act)
        self._residual_gemm("down", self.act, w_down, hmid, h_out)

    def gdn_layer(self, index, h_in, gamma_in, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias,
                  gamma_gdn, w_out, gamma_post, w_gate, w_up, w_down):
        slot = GDN_SLOT[index]
        h_out = self.out_hidden(h_in)
        w_in = self.pack(f"layer{index}.in", (w_qkv, w_z, w_b, w_a), GDN_IN_PAD)
        self.k_norm(h_in, gamma_in, self.x)
        self.k_gdn_in(self.x, w_in, self.proj)
        self.gdn_ba(index, w_b, w_a)
        self.gdn.prep(self.proj[:, :GDN_IN], conv_w.view(CONV, WINDOW + 1), a_log, dt_bias,
                      self.q, self.k, self.v, self.g, self.beta, self.conv[slot])
        z = self.proj[:, CONV:CONV + HV * DV].view(self.T, HV, DV)
        self.gdn.core(self.q, self.k, self.v, self.g, self.beta, z, gamma_gdn, self.y, self.rec[slot])
        self._residual_gemm("o_proj", self.y, w_out, h_in, self.hmid)
        self.mlp(self.hmid, h_out, gamma_post, w_gate, w_up, w_down)
        return h_out, self.conv[slot], self.rec[slot]

    def gdn_ba(self, index, w_b, w_a):
        """Overwrite the b|a columns of self.proj (from self.x) in cuBLAS's split-K order."""
        w_ba = self.pack(f"layer{index}.ba", (w_b, w_a))
        self.k_ba(self.x, w_ba, self.ba_part)
        self.k_ba_red(self.ba_part, self.proj)

    def attn_layer(self, index, h_in, cos, sin, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k, w_o,
                   gamma_post, w_gate, w_up, w_down):
        slot = ATT_SLOT[index]
        h_out = self.out_hidden(h_in)
        w_in = self.pack(f"layer{index}.in", (w_q, w_k, w_v))
        self.k_norm(h_in, gamma_in, self.x)
        self.k_att_in(self.x, w_in, self.qkv)
        self.att.prep(self.qkv, gamma_q, gamma_k, cos, sin, self.qa, self.kc[slot], self.vc[slot])
        self.att.core(self.qa, self.kc[slot], self.vc[slot], self.qkv, self.y)
        self._residual_gemm("o_proj", self.y, w_o, h_in, self.hmid)
        self.mlp(self.hmid, h_out, gamma_post, w_gate, w_up, w_down)
        return h_out, self.kc[slot], self.vc[slot]

    def embed(self, ids, table):
        self.k_embed(ids, table, self.h[0])
        return self.h[0]

    def head(self, h, gamma_final, w_head):
        self.k_norm1(h[self.T - 1:self.T], gamma_final, self.xlast)
        self.k_head(self.xlast.view(H), w_head, self.logits.view(V))
        return self.logits

    # ----------------------------------------------------------------- graph
    def set_inputs(self, token_ids, cos, sin):
        self.ids.copy_(token_ids)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def capture(self, body):
        self.stream = torch.cuda.Stream(self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            body()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.graph = g

    def outputs(self):
        return self.logits, self.kc, self.vc, self.conv, self.rec


_ENGINES: dict[tuple[int, int], Engine] = {}
# Packed input projections, shared by the engines of every length (one copy per layer).
_PACKED: dict[str, tuple] = {}


def engine_for(T: int, device=None) -> Engine:
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    key = (T, device.index or 0)
    if key not in _ENGINES:
        _ENGINES[key] = Engine(T, device)
    return _ENGINES[key]
