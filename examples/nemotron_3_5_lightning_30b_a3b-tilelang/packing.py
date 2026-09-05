"""Where every declared weight sits once the twin has packed it.

A kernel cannot take 401 pointers, so the twin holds one flat tensor per weight
class -- layers stacked on the leading axis -- and addresses any of them by
(layer, row). That layout is a property of the model, not of any one backend,
so it lives here: `mega_kernel.py` reads it and so does the handwritten CUDA
path, and neither can drift from the other.

Nothing here imports a backend. The twin's `load` runs before an
implementation is chosen, and making it import one would make every
implementation need every backend installed.
"""
from __future__ import annotations

import torch

import model

KINDS = model.LAYER_KINDS
NL = len(KINDS)

H, V, EPS = model.H, model.V, model.EPS
MH, MD, SS, NG, MI, CONV, KER, WIN, PROJ = (model.MH, model.MD, model.SS, model.NG,
                                            model.MI, model.CONV, model.KER, model.WIN,
                                            model.PROJ)
GRP, HPG, DTMIN = model.GRP, model.HPG, model.DTMIN
HQ, HKV, DH, QP, KVP, GQA, QSCALE = (model.HQ, model.HKV, model.DH, model.QP,
                                     model.KVP, model.GQA, model.QSCALE)
E, KTOP, I, IS, RSCALE = model.E, model.K, model.I, model.IS, model.RSCALE
KINDS = model.LAYER_KINDS

_KIND_ID = {"linear_attention": 0, "full_attention": 1, "moe": 2}
_LAYER_META, _seen = [], {0: 0, 1: 0, 2: 0}
for _k in KINDS:
    _LAYER_META.append((_KIND_ID[_k], _seen[_KIND_ID[_k]]))
    _seen[_KIND_ID[_k]] += 1
N_MAMBA, N_ATTN, N_MOE = _seen[0], _seen[1], _seen[2]
PACK = {
    "win": (N_MAMBA, PROJ, H), "wout": (N_MAMBA, H, MI),
    "convw": (N_MAMBA, CONV, KER), "convb": (N_MAMBA, CONV), "ggdn": (N_MAMBA, MI),
    "wqkv": (N_ATTN, QP + 2 * KVP, H), "wo": (N_ATTN, H, QP),
    "wrt": (N_MOE, E, H), "wup": (N_MOE, E, I, H), "wdn": (N_MOE, E, H, I),
    "wsu": (N_MOE, IS, H), "wsd": (N_MOE, H, IS),
    "gam": (NL, H), "table": (V, H), "whead": (V, H),
    "mscal": (N_MAMBA, 3, MH), "gf": (H,),
}
PACK_F32 = {"eb": (N_MOE, E)}
PACK_ORDER = list(PACK) + list(PACK_F32)
RING = 4


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def bf(x):
    """Round to bf16 and go on in f32, the way a bf16 op does."""
    return T.Cast("float32", T.Cast("bfloat16", x))


class Packed:
    """Every weight of one class in one flat tensor."""

    def __init__(self, device):
        self.shape = dict(PACK) | dict(PACK_F32)
        self.t = {n: torch.zeros(_numel(s), dtype=torch.bfloat16, device=device)
                  for n, s in PACK.items()}
        self.t.update({n: torch.zeros(_numel(s), dtype=torch.float32, device=device)
                       for n, s in PACK_F32.items()})

    def view(self, name):
        return self.t[name].view(self.shape[name])

    def flat_order(self):
        return [self.t[n] for n in PACK_ORDER]


def pack_into(packed: Packed, name: str, value: torch.Tensor):
    """Copy one declared weight into its slot; return the view standing for it."""
    def put(key, *idx):
        dst = packed.view(key)
        for j in idx:
            dst = dst[j]
        dst.copy_(value.reshape(dst.shape).to(dst.dtype))
        return dst

    if name in ("table", "w_head"):
        return put({"table": "table", "w_head": "whead"}[name])
    if name == "gamma_final":
        return put("gf")
    layer = int(name.split("_")[0][1:])
    kid, at = _LAYER_META[layer]
    tail = name[name.index("_") + 1:]
    if tail == "gamma":
        return put("gam", layer)
    if kid == 0:
        simple = {"w_in": "win", "w_out": "wout", "conv_w": "convw",
                  "conv_b": "convb", "gamma_gdn": "ggdn"}
        if tail in simple:
            return put(simple[tail], at)
        return put("mscal", at, {"a_log": 0, "dt_bias": 1, "d_skip": 2}[tail])
    if kid == 1:
        if tail == "w_o":
            return put("wo", at)
        off = {"w_q": 0, "w_k": QP, "w_v": QP + KVP}[tail]
        dst = packed.view("wqkv")[at, off:off + value.shape[0]]
        dst.copy_(value)
        return dst
    simple = {"w_router": "wrt", "w_up": "wup", "w_down": "wdn",
              "w_sh_up": "wsu", "w_sh_down": "wsd", "e_bias": "eb"}
    return put(simple[tail], at)


