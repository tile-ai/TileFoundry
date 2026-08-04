"""Which side of the attention disagreement is closer to truth?

`check` reports that the twin and the authored reference differ, and says plainly
that it cannot say which is closer -- "establishing accuracy needs an independent
high-precision reference, which check does not run." This runs one: the same real
activations and the same real weights through the same math in float64.

The disagreement is not an accident. The reference multiplies `1/sqrt(head_dim)`
onto `q` in bf16 before the dot product, rounding every entry a second time, and
the exponential downstream magnifies it. The kernels apply the factor to the
finished f32 dot instead, which is also what Hugging Face does. This script is the
evidence for calling that an improvement rather than a deviation.
"""
import argparse
from pathlib import Path

import torch

from engine import _load_module, default_ref_dir


def f64_attention(hidden, w, cos, sin, pos, k_cache, v_cache, scale, cfg):
    """Reference math in f64: no intermediate lands in bf16 anywhere."""
    H = cfg.hidden_size
    HQ, HKV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = HQ // HKV
    eps = cfg.rms_norm_eps
    f = torch.float64

    x = hidden.reshape(H).to(f)
    x = x * torch.rsqrt(x.pow(2).mean() + eps) * w["gamma_in"].to(f)
    q = (x @ w["w_q"][0].to(f)).reshape(HQ, D)
    k = (x @ w["w_k"][0].to(f)).reshape(HKV, D)
    v = (x @ w["w_v"][0].to(f)).reshape(HKV, D)

    def hnorm(t, g):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * g.to(f)

    q = hnorm(q, w["gamma_q"])
    k = hnorm(k, w["gamma_k"])

    def rope(t):
        c, s = cos[pos].to(f), sin[pos].to(f)
        half = torch.cat([-t[:, D // 2:], t[:, : D // 2]], dim=-1)
        return t * c + half * s

    q, k = rope(q), rope(k)
    ctx = int(k_cache.shape[1])
    kk = torch.cat([k_cache[0].to(f), k.unsqueeze(0)], dim=0)      # (ctx+1, HKV, D)
    vv = torch.cat([v_cache[0].to(f), v.unsqueeze(0)], dim=0)
    kk = kk.repeat_interleave(G, dim=1)                            # (ctx+1, HQ, D)
    vv = vv.repeat_interleave(G, dim=1)
    sc = (q.unsqueeze(0) * kk).sum(-1) * float(scale.reshape(-1)[0])
    p = torch.softmax(sc, dim=0)
    attn = (p.unsqueeze(-1) * vv).sum(0).reshape(HQ * D)
    return attn @ w["w_o"][0].to(f)


def err(got, truth):
    g, t = got.reshape(-1).to(torch.float64), truth.reshape(-1)
    return ((g - t).norm() / t.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=Path("real_inputs"))
    ap.add_argument("--ckpt", default="../prepared")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    from tilefoundry.runtime import SafetensorsResource
    import twin

    ref = twin.ref
    cfg = ref.config
    dev = "cuda:0"
    acts = {n: torch.load(args.real / f"{n}.pt").to(dev)
            for n in ("hidden", "cos_cache", "sin_cache", "pos_ids",
                      "k_cache", "v_cache", "scale")}

    loaded = ref.Qwen3_1_7B.load(SafetensorsResource(str(args.ckpt), device=dev))
    lay = getattr(loaded, f"layer{args.layer}")
    w = lay.constants

    # both sides take activations alone; each fills its own weights from its own
    # reading of the same checkpoint, which is what makes the comparison fair
    call = [acts[n] for n in ("hidden", "cos_cache", "sin_cache", "pos_ids",
                              "k_cache", "v_cache", "scale")]

    truth = f64_attention(
        acts["hidden"], w, acts["cos_cache"], acts["sin_cache"],
        int(acts["pos_ids"][0]), acts["k_cache"], acts["v_cache"], acts["scale"], cfg,
    )

    ref_out = lay.self_attention(*call)[0]
    tw = twin.LayerTwin(ir=ref.Qwen3_1_7B_DecoderLayer)
    tw.load(SafetensorsResource(str(args.ckpt), device=dev).subtree(f"layer{args.layer}"))
    mine = tw.self_attention(*call)[0]

    e_ref, e_mine = err(ref_out, truth), err(mine, truth)
    print(f"independent f64 reference, layer {args.layer}, "
          f"ctx_len {int(acts['k_cache'].shape[1])}\n")
    print(f"  authored HIR (evaluator)  rel_l2 vs f64 = {e_ref:.3e}")
    print(f"  TileLang twin             rel_l2 vs f64 = {e_mine:.3e}")
    print(f"  twin/reference error ratio              = {e_mine / e_ref:.3f}")
    verdict = ("the twin is CLOSER to truth" if e_mine < e_ref
               else "the reference is closer to truth")
    print(f"\n  -> {verdict}")


if __name__ == "__main__":
    main()
