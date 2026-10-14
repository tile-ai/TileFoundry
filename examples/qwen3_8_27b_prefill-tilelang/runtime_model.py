"""Runtime twin of model.py on the tilelang backend: every model.py function is
one kernel or one tight group of them (evidence/STEP3.md), launched from here.

    function (model.py)          kernel(s) (kernels/)
    Qwen3_8_27B.embed            elementwise.embed
    Qwen3_8_27B.final            attention.final      (closing RMSNorm + lm_head GEMV)
    layer.input_norm/post_norm   elementwise.rmsnorm
    mixer.in_proj                gemm  plain          (packed [qkv | z | b | a | 0] weight)
    mixer.prep                   elementwise.prep     (conv, SiLU, q/k l2norm, g, beta)
    mixer.delta_rule             gdn.intra + gdn.recur_out
    mixer.gated_norm             elementwise.gated_norm
    mixer.out_proj               gemm  residual
    mixer.qkv                    gemm  qk_norm_rope   (packed [q|gate per head | k | v] weight)
    mixer.attention              attention.attention_tir (causal flash + output gate, warp-specialized, FA3 ping-pong)
    mixer.o_proj                 gemm  residual
    mlp.gate_up                  gemm  swiglu         (packed: gate / up rows interleaved by 128)
    mlp.down                     gemm  residual

Packed weights: in_proj, qkv and gate_up read one weight assembled from the
checkpoint's tensors (pack_* below = scheduled/gemm.py's pack_* funcs). A
pack reads only weights, so it runs once, on the function's first call, and
its result is held: a CUDA graph replays raw pointers, so every weight it
reads must stay alive. A GEMM with a split tail is two launches, the main
kernel and gemm_tail_kernel (kernels/gemm.py).

A function whose inputs are the views its producer returned (the forward
path) hands the producer's whole buffer to the kernel; one called with
independent tensors (`tilefoundry check` reads them from files) packs them
into that buffer first, then runs the same kernel.
"""
from __future__ import annotations

import torch

import model as M
from kernels import attention as A
from kernels import elementwise as E
from kernels import gdn as GD
from kernels import gemm as G
from kernels import pack as PK
from tilefoundry.dsl import ModuleList
from tilefoundry.runtime import runtime_func, runtime_module

H, I, V = M.H, M.I, M.V
HQ, HKV, D = M.HQ, M.HKV, M.D
HK, HV, DK, DV = M.HK, M.HV, M.DK, M.DV
CONV, VAL, KEY = M.CONV, M.VAL, M.KEY
P_USED = CONV + VAL + 2 * HV       # qkv | z | b | a = 16480 columns
P_W = 16640                        # in_proj weight rows and output width: 65 whole 256-wide tiles
QKV_W = HQ * 2 * D + 2 * HKV * D   # qkv output width: q|gate per head | k | v
BF = torch.bfloat16


def pack_gate_up(w_gate, w_up):
    """[gate rows 128 j .. | up rows 128 j ..] per 256-row tile (kernels/pack.py)."""
    w = torch.empty(2 * I, H, device=w_gate.device, dtype=BF)
    PK.compiled("gate_up")(w_gate, w_up, w)
    return w


def pack_in_proj(w_qkv, w_z, w_b, w_a):
    """[qkv | z | b | a | zero rows] -- padded to whole 256-row tiles."""
    w = torch.empty(P_W, H, device=w_qkv.device, dtype=BF)
    PK.compiled("in_proj")(w_qkv, w_z, w_b, w_a, w)
    return w


def pack_qkv(w_q, w_k, w_v):
    """[q|gate per head | k | v]."""
    w = torch.empty(QKV_W, H, device=w_q.device, dtype=BF)
    PK.compiled("qkv")(w_q, w_k, w_v, w)
    return w


def _held(self, key, build):
    """A weight rearranged once and kept alive for the instance's lifetime."""
    store = self.__dict__.setdefault("_held", {})
    if key not in store:
        store[key] = build()
    return store[key]


def _n(x):
    return x.shape[1]


def _base_of(view, width):
    """The (N, width) buffer `view` was cut from, if it is a column view of one."""
    n = view.shape[1]
    if view.stride(-1) != 1 or view.dim() < 2:
        return None
    st = view.untyped_storage()
    base = torch.empty(0, dtype=view.dtype, device=view.device).set_(st, 0, (n, width), (width, 1))
    return base


@runtime_module(M.MLP)
class MLPTwin:
    @runtime_func
    def gate_up(self, hn, w_gate, w_up):
        n = _n(hn)
        w = _held(self, "w_gu", lambda: pack_gate_up(w_gate, w_up))
        _held(self, "raw", lambda: (w_gate, w_up))
        out = torch.empty(1, n, I, device=hn.device, dtype=BF)
        G.compile_gemm(n, 2 * I, H, "swiglu")(hn.view(n, H), w, out.view(n, I))
        return out

    @runtime_func
    def down(self, act, hidden, w_down):
        n = _n(act)
        _held(self, "w_down", lambda: w_down)
        out = torch.empty(1, n, H, device=act.device, dtype=BF)
        G.compile_gemm(n, H, I, "residual")(act.view(n, I), w_down, hidden.view(n, H), out.view(n, H))
        return out


@runtime_module(M.GatedDeltaNet)
class GatedDeltaNetTwin:
    @runtime_func
    def in_proj(self, hn, w_qkv, w_z, w_b, w_a):
        n = _n(hn)
        w = _held(self, "w_in", lambda: pack_in_proj(w_qkv, w_z, w_b, w_a))
        p = torch.empty(n, P_W, device=hn.device, dtype=BF)
        G.compile_gemm(n, P_W, H, "plain")(hn.view(n, H), w, p)
        o = 0
        outs = []
        for width in (CONV, VAL, HV, HV):
            outs.append(p[:, o:o + width].unsqueeze(0))
            o += width
        return tuple(outs)

    def _packed(self, qkv, b, a):
        """The in_proj buffer qkv / b / a were cut from, or a fresh packing of them."""
        n = _n(qkv)
        if (qkv.stride(1) == P_W and b.stride(1) == P_W and a.stride(1) == P_W
                and qkv.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
                and qkv.storage_offset() == 0 and b.storage_offset() == CONV + VAL
                and a.storage_offset() == CONV + VAL + HV):
            return _base_of(qkv, P_W)
        p = torch.zeros(n, P_W, device=qkv.device, dtype=BF)
        p[:, :CONV] = qkv[0]
        p[:, CONV + VAL:CONV + VAL + HV] = b[0]
        p[:, CONV + VAL + HV:P_USED] = a[0]
        return p

    @runtime_func
    def prep(self, qkv, b, a, conv_w, a_log, dt_bias):
        n = _n(qkv)
        p = self._packed(qkv, b, a)
        cw = _held(self, "conv_w", lambda: conv_w.reshape(CONV, 4).contiguous())
        _held(self, "gates", lambda: (a_log, dt_bias))
        dev = qkv.device
        q = torch.empty(1, n, HK, DK, device=dev, dtype=BF)
        k = torch.empty(1, n, HK, DK, device=dev, dtype=BF)
        v = torch.empty(1, n, HV, DV, device=dev, dtype=BF)
        g = torch.empty(1, n, HV, device=dev, dtype=torch.float32)
        beta = torch.empty(1, n, HV, device=dev, dtype=BF)
        E.compiled("prep", n, QS=P_W)(p, cw, a_log, dt_bias, q[0], k[0], v[0], g[0], beta[0])
        return q, k, v, g, beta

    @runtime_func
    def delta_rule(self, q, k, v, g, beta):
        n = _n(q)
        dev = q.device
        wu = torch.empty(n, HV * GD.WUP, device=dev, dtype=BF)   # [w | u | P row] per (token, head)
        gc = torch.empty(HV, n, device=dev, dtype=torch.float32)
        o = torch.empty(1, n, HV, DV, device=dev, dtype=BF)
        # tir/delta_rule_intra_*.py read their masks' row / column numbers from this table
        iota = _held(self, "iota_ch", lambda: torch.arange(GD.CH, device=dev, dtype=torch.int32))
        GD.compiled("intra", n)(q[0], k[0], v[0], g[0], beta[0], iota, wu, gc)
        GD.compiled("recur_out", n)(q[0], k[0], wu, gc, o[0])
        return o

    @runtime_func
    def gated_norm(self, core, z, gamma_gdn):
        n = _n(core)
        _held(self, "gamma_gdn", lambda: gamma_gdn)
        if z.stride(1) == P_W and z.storage_offset() == CONV:
            p = _base_of(z, P_W)
        else:
            p = torch.zeros(n, P_W, device=z.device, dtype=BF)
            p[:, CONV:CONV + VAL] = z[0]
        out = torch.empty(1, n, VAL, device=core.device, dtype=BF)
        E.compiled("gated_norm", n, ZS=P_W)(core.contiguous().view(-1, DV), p, gamma_gdn, out.view(-1, DV))
        return out

    @runtime_func
    def out_proj(self, gated, hidden, w_out):
        n = _n(gated)
        _held(self, "w_out", lambda: w_out)
        out = torch.empty(1, n, H, device=gated.device, dtype=BF)
        G.compile_gemm(n, H, VAL, "residual")(gated.view(n, VAL), w_out, hidden.view(n, H), out.view(n, H))
        return out


@runtime_module(M.Attention)
class AttentionTwin:
    @runtime_func
    def qkv(self, hn, cos, sin, w_q, w_k, w_v, gamma_q, gamma_k):
        n = _n(hn)
        w = _held(self, "w_qkv", lambda: pack_qkv(w_q, w_k, w_v))
        _held(self, "gammas", lambda: (gamma_q, gamma_k))
        buf = torch.empty(n, QKV_W, device=hn.device, dtype=BF)
        G.compile_gemm(n, QKV_W, H, "qk_norm_rope")(hn.view(n, H), w, gamma_q, gamma_k,
                                                   cos.contiguous(), sin.contiguous(), buf)
        qg = buf[:, :HQ * 2 * D].view(1, n, HQ, 2 * D)
        q = qg[..., :D]
        gate = qg[..., D:]
        k = buf[:, HQ * 2 * D:HQ * 2 * D + HKV * D].view(1, n, HKV, D)
        v = buf[:, HQ * 2 * D + HKV * D:].view(1, n, HKV, D)
        return q, gate, k, v

    @runtime_func
    def attention(self, q, k, v, gate):
        n = _n(q)
        same = (q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
                == v.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr())
        if (same and q.storage_offset() == 0 and gate.storage_offset() == D
                and k.storage_offset() == HQ * 2 * D and v.storage_offset() == HQ * 2 * D + HKV * D
                and q.stride(1) == QKV_W):
            buf = _base_of(q, QKV_W)
        else:
            buf = torch.empty(n, QKV_W, device=q.device, dtype=BF)
            qg = buf[:, :HQ * 2 * D].view(n, HQ, 2 * D)
            qg[..., :D] = q[0]
            qg[..., D:] = gate[0]
            buf[:, HQ * 2 * D:HQ * 2 * D + HKV * D] = k[0].reshape(n, -1)
            buf[:, HQ * 2 * D + HKV * D:] = v[0].reshape(n, -1)
        out = torch.empty(1, n, HQ * D, device=q.device, dtype=BF)
        if n > 128:
            # tir/attention_{n}.py reads its mask's row / column numbers from this table
            iota = _held(self, "iota", lambda: torch.arange(128, device=q.device, dtype=torch.int32))
            A.compiled("attention_tir", n)(buf, iota, out.view(n, HQ * D))
        else:
            # one 128-row query block: nothing to ping-pong, and TileLang
            # 0.1.15 cannot lower attention_pp's syncs at M = 128
            A.compiled("attention_ws", n)(buf, out.view(n, HQ * D))
        return out

    @runtime_func
    def o_proj(self, gated, hidden, w_o):
        n = _n(gated)
        _held(self, "w_o", lambda: w_o)
        out = torch.empty(1, n, H, device=gated.device, dtype=BF)
        G.compile_gemm(n, H, HQ * D, "residual")(gated.view(n, HQ * D), w_o, hidden.view(n, H), out.view(n, H))
        return out


def _norm(self, hidden, gamma, key):
    n = _n(hidden)
    _held(self, key, lambda: gamma)
    out = torch.empty_like(hidden)
    E.compiled("rmsnorm", n)(hidden.view(n, H), gamma, out.view(n, H))
    return out


@runtime_module(M.LinearAttentionLayer)
class LinearAttentionLayerTwin:
    mixer = GatedDeltaNetTwin
    mlp = MLPTwin

    @runtime_func
    def input_norm(self, hidden, gamma_in):
        return _norm(self, hidden, gamma_in, "gamma_in")

    @runtime_func
    def post_norm(self, hidden, gamma_post):
        return _norm(self, hidden, gamma_post, "gamma_post")


@runtime_module(M.FullAttentionLayer)
class FullAttentionLayerTwin:
    mixer = AttentionTwin
    mlp = MLPTwin

    @runtime_func
    def input_norm(self, hidden, gamma_in):
        return _norm(self, hidden, gamma_in, "gamma_in")

    @runtime_func
    def post_norm(self, hidden, gamma_post):
        return _norm(self, hidden, gamma_post, "gamma_post")


def _root_class():
    twins = {"linear_attention": LinearAttentionLayerTwin, "full_attention": FullAttentionLayerTwin}
    body = {"layers": ModuleList(tuple(twins[kind] for kind in M.config["layer_types"]))}

    def embed(self, table, token_ids):
        n = token_ids.shape[0]
        _held(self, "table", lambda: table)
        out = torch.empty(1, n, H, device=token_ids.device, dtype=BF)
        E.compiled("embed", n)(token_ids, table, out.view(n, H))
        return out

    def final(self, hidden, last, gamma_final, w_head):
        n = _n(hidden)
        _held(self, "head", lambda: (gamma_final, w_head))
        out = torch.empty(1, V, device=hidden.device, dtype=BF)
        A.compiled("final", n)(hidden.view(n, H), last, gamma_final, w_head, out)
        return out

    body["embed"] = runtime_func(embed)
    body["final"] = runtime_func(final)
    body["__doc__"] = "The whole prefill: model.py's root and every layer, on tilelang kernels."
    return runtime_module(M.Qwen3_8_27B)(type("Qwen3_8_27BTwin", (), body))


Qwen3_8_27BTwin = _root_class()
