"""The tilelang runtime module: the twin of model.py's Modules, one runtime_func per
authored function, every one of them backed by tilelang kernels (runtime/kernels).

`check runtime/twin.py:Qwen3_8_27BTwin` compares it against the evaluator running
model.py. The fast path is `prefill`: on the first call for a length it runs the
body once (compiles, packs the input projections), captures the same body in a CUDA
graph, and from then on copies the three inputs into the graph's static buffers and
replays it.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model  # noqa: E402
from runtime import engine as eng  # noqa: E402
from runtime.kernels import small  # noqa: E402
from runtime.kernels.gemm import GemmConfig, gemm_kernel  # noqa: E402
from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

_STORE = GemmConfig(BM=128, BN=128, BK=64, stages=4)
_GDN48: dict[int, object] = {}


def _T(x) -> int:
    return x.shape[1] if x.dim() >= 3 else x.shape[0]


def _engine(T: int, like: torch.Tensor):
    return eng.engine_for(T, like.device)


def _index(twin) -> int:
    return int(twin.name.removeprefix("layer"))


def _swiglu(x, w_gate, w_up, w_down):
    T = _T(x)
    e = _engine(T, x)
    e.k_gu(x.reshape(T, eng.H), w_gate, w_up, e.act)
    out = torch.empty(1, T, eng.H, device=x.device, dtype=x.dtype)
    gemm_kernel(T, eng.H, eng.I, _STORE, "store")(e.act, w_down, out.view(T, eng.H))
    return out


@runtime_module(model.Qwen3_8DeltaNetLayer)
class DeltaNetLayerTwin:
    @runtime_func
    def swiglu(self, x, w_gate, w_up, w_down):
        return _swiglu(x, w_gate, w_up, w_down)

    @runtime_func
    def causal_conv(self, mixed, conv_w):
        T = _T(mixed)
        out = torch.empty_like(mixed)
        small.conv_silu_kernel(T, eng.CONV)(mixed.view(T, eng.CONV), conv_w.view(eng.CONV, eng.WINDOW + 1),
                                             out.view(T, eng.CONV))
        return out

    @runtime_func
    def l2_normalise(self, x):
        T = _T(x)
        out = torch.empty_like(x)
        small.l2norm_kernel(T, x.shape[2], x.shape[3])(x.view(T, x.shape[2], x.shape[3]),
                                                        out.view(T, x.shape[2], x.shape[3]))
        return out

    @runtime_func
    def delta_rule(self, q, k, v, g, beta):
        # A standalone call may hand 48 independent key heads; the fast path hands 16.
        from runtime.kernels.gdn import GdnKernels

        T = _T(q)
        if T not in _GDN48:
            _GDN48[T] = GdnKernels(T, key_heads=eng.HV)
        read = torch.empty(1, T, eng.HV, eng.DV, device=q.device, dtype=q.dtype)
        state = torch.empty(1, eng.HV, eng.DK, eng.DV, device=q.device, dtype=torch.float32)
        _GDN48[T].core(q.view(T, eng.HV, eng.DK), k.view(T, eng.HV, eng.DK), v.view(T, eng.HV, eng.DV),
                       g.view(T, eng.HV), beta.view(T, eng.HV), None, None,
                       read.view(T, eng.HV * eng.DV), state.view(eng.HV, eng.DK, eng.DV))
        return read, state

    @runtime_func
    def gated_delta_net(self, x, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, gamma_gdn, w_out):
        T = _T(x)
        e = _engine(T, x)
        w_in = e.pack(f"{self.name}.in", (w_qkv, w_z, w_b, w_a), eng.GDN_IN_PAD)
        e.k_gdn_in(x.view(T, eng.H), w_in, e.proj)
        e.x.copy_(x.view(T, eng.H))
        e.gdn_ba(_index(self), w_b, w_a)
        window = torch.empty(1, eng.CONV, eng.WINDOW, device=x.device, dtype=x.dtype)
        state = torch.empty(1, eng.HV, eng.DK, eng.DV, device=x.device, dtype=torch.float32)
        e.gdn.prep(e.proj[:, :eng.GDN_IN], conv_w.view(eng.CONV, eng.WINDOW + 1), a_log, dt_bias,
                   e.q, e.k, e.v, e.g, e.beta, window[0])
        z = e.proj[:, eng.CONV:eng.CONV + eng.HV * eng.DV].view(T, eng.HV, eng.DV)
        e.gdn.core(e.q, e.k, e.v, e.g, e.beta, z, gamma_gdn, e.y, state[0])
        out = torch.empty(1, T, eng.H, device=x.device, dtype=x.dtype)
        gemm_kernel(T, eng.H, eng.HV * eng.DV, _STORE, "store")(e.y, w_out, out.view(T, eng.H))
        return out, window, state

    @runtime_func
    def layer(self, hidden, gamma_in, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, gamma_gdn, w_out,
              gamma_post, w_gate, w_up, w_down):
        T = _T(hidden)
        e = _engine(T, hidden)
        h, window, state = e.gdn_layer(_index(self), hidden.view(T, eng.H), gamma_in, w_qkv, w_z, w_b, w_a,
                                       conv_w, a_log, dt_bias, gamma_gdn, w_out, gamma_post, w_gate, w_up, w_down)
        return h.view(1, T, eng.H), window.unsqueeze(0), state.unsqueeze(0)


@runtime_module(model.Qwen3_8AttentionLayer)
class AttentionLayerTwin:
    @runtime_func
    def swiglu(self, x, w_gate, w_up, w_down):
        return _swiglu(x, w_gate, w_up, w_down)

    @runtime_func
    def partial_rope(self, x, cos, sin):
        T = _T(x)
        out = torch.empty_like(x)
        small.rope_kernel(T, x.shape[2], x.shape[3])(x[0], cos, sin, out[0])
        return out

    @runtime_func
    def partial_rope_kv(self, x, cos, sin):
        T = _T(x)
        out = torch.empty_like(x)
        small.rope_kernel(T, x.shape[2], x.shape[3])(x[0], cos, sin, out[0])
        return out

    @runtime_func
    def attend(self, q, k, v):
        # The core kernel multiplies by bf16(sigmoid(gate)); a gate of 1e4 makes that
        # factor exactly 1.0, so the product is the attention output unchanged.
        T = _T(q)
        e = _engine(T, q)
        gate = torch.full((T, eng.ATT_IN), 1e4, device=q.device, dtype=q.dtype)
        y = torch.empty(T, eng.HQ * eng.D, device=q.device, dtype=q.dtype)
        e.att.core(q[0], k[0], v[0], gate, y)
        return y.view(1, T, eng.HQ, eng.D)

    @runtime_func
    def gated_attention(self, x, cos, sin, w_q, w_k, w_v, gamma_q, gamma_k, w_o):
        T = _T(x)
        e = _engine(T, x)
        w_in = e.pack(f"{self.name}.in", (w_q, w_k, w_v))
        e.k_att_in(x.view(T, eng.H), w_in, e.qkv)
        kr = torch.empty(1, T, eng.HKV, eng.D, device=x.device, dtype=x.dtype)
        vv = torch.empty_like(kr)
        e.att.prep(e.qkv, gamma_q, gamma_k, cos, sin, e.qa, kr[0], vv[0])
        e.att.core(e.qa, kr[0], vv[0], e.qkv, e.y)
        out = torch.empty(1, T, eng.H, device=x.device, dtype=x.dtype)
        gemm_kernel(T, eng.H, eng.HQ * eng.D, _STORE, "store")(e.y, w_o, out.view(T, eng.H))
        return out, kr, vv

    @runtime_func
    def layer(self, hidden, cos, sin, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k, w_o,
              gamma_post, w_gate, w_up, w_down):
        T = _T(hidden)
        e = _engine(T, hidden)
        h, k, v = e.attn_layer(_index(self), hidden.view(T, eng.H), cos, sin, gamma_in, w_q, w_k, w_v,
                               gamma_q, gamma_k, w_o, gamma_post, w_gate, w_up, w_down)
        return h.view(1, T, eng.H), k.unsqueeze(0), v.unsqueeze(0)


def _root_body(self, e, table, gamma_final, w_head):
    h = e.embed(e.ids, table)
    for i, kind in enumerate(eng.KINDS):
        layer = getattr(self, f"layer{i}")
        if kind == "linear_attention":
            out, _, _ = layer.layer(h.view(1, e.T, eng.H))
        else:
            out, _, _ = layer.layer(h.view(1, e.T, eng.H), e.cos, e.sin)
        h = out.view(e.T, eng.H)
    e.head(h, gamma_final, w_head)


def _embed(self, token_ids, table):
    T = token_ids.shape[0]
    out = torch.empty(1, T, eng.H, device=table.device, dtype=table.dtype)
    small.embed_kernel(T, eng.V, eng.H)(token_ids, table, out.view(T, eng.H))
    return out


def _final_norm(self, hidden, gamma_final):
    T = _T(hidden)
    out = torch.empty_like(hidden)
    small.rmsnorm_kernel(T, eng.H)(hidden.view(T, eng.H), gamma_final, out.view(T, eng.H))
    return out


def _lm_head(self, hidden, w_head):
    T = _T(hidden)
    out = torch.empty(1, eng.V, device=hidden.device, dtype=hidden.dtype)
    small.gemv_kernel(eng.V, eng.H)(hidden.view(T, eng.H)[T - 1], w_head, out.view(eng.V))
    return out


def _prefill(self, token_ids, cos, sin, table, gamma_final, w_head):
    T = token_ids.shape[0]
    e = _engine(T, table)
    e.set_inputs(token_ids, cos, sin)
    if e.graph is None:
        _root_body(self, e, table, gamma_final, w_head)  # eager once: compile, pack
        torch.cuda.synchronize(e.device)
        e.capture(lambda: _root_body(self, e, table, gamma_final, w_head))
        e.set_inputs(token_ids, cos, sin)
    e.graph.replay()
    return e.outputs()


def _make_root():
    ns = {
        "embed": runtime_func(_embed),
        "final_norm": runtime_func(_final_norm),
        "lm_head": runtime_func(_lm_head),
        "prefill": runtime_func(_prefill),
        "__doc__": "Twin of model.Qwen3_8_27B.",
        "__module__": __name__,
    }
    for i, kind in enumerate(eng.KINDS):
        ns[f"layer{i}"] = DeltaNetLayerTwin if kind == "linear_attention" else AttentionLayerTwin
    return runtime_module(model.Qwen3_8_27B)(type("Qwen3_8_27BTwin", (), ns))


Qwen3_8_27BTwin = _make_root()
