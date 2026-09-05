"""Compare the handwritten MoE layer against the op-by-op reference, stage by stage.

Same idea as `check_layer.py` for Mamba-2: the whole-step check cannot say which
part of a layer disagrees, so this runs the router, the six routed experts and
the shared one against the torch expressions `runtime_model._ops_step` uses.

    python check_moe.py --layer 1 [--bench 200]
"""
from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F

import model
import paths

H, E, K, I, IS, EPS, RSCALE = (model.H, model.E, model.K, model.I, model.IS,
                               model.EPS, model.RSCALE)
BF16 = torch.bfloat16
ULP = 2.0 ** -9


def reference(h, gamma, w_router, e_bias, w_up, w_down, w_sh_up, w_sh_down):
    """The MoE branch of `runtime_model._ops_step`, operation for operation."""
    hh = h.reshape(1, 1, H)
    xf = hh.float()
    nz = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(BF16)
    h2 = (nz * gamma.reshape(1, 1, -1)).reshape(1, H)

    lg = h2.float() @ w_router.float().t()
    sig = torch.sigmoid(lg)
    ch = sig + e_bias.reshape(1, E)
    _tv, ti = torch.topk(ch, K, dim=-1, sorted=False)
    flat = ti.reshape(K)
    pick = sig.reshape(E)[flat].reshape(1, K)
    gw = (pick / (pick.sum(-1, keepdim=True) + 1e-20)) * RSCALE

    total = None
    for j, e in enumerate(flat.tolist()):
        mid = torch.square(torch.relu(h2 @ w_up[e].t()))
        r = (mid @ w_down[e].t()).reshape(H).float() * gw[0, j]
        total = r if total is None else total + r
    smid = torch.square(torch.relu(h2 @ w_sh_up.t()))
    sh = (smid @ w_sh_down.t()).reshape(H)
    mix = (total.to(BF16) + sh).reshape(1, 1, H)
    return {"h2": h2.reshape(-1), "idx": flat, "gw": gw.reshape(-1),
            "acc": total, "smid": smid.reshape(-1), "h_out": (hh + mix).reshape(-1)}


def rel_l2(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    d = b.norm().item()
    return (a - b).norm().item() / (d if d else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()

    import runtime_model as R  # noqa: PLC0415
    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415

    dev = "cuda:0"
    twin = R.Nemotron35Lightning30BA3BRuntime()
    R.set_impl("ops")
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=dev))

    params = model.Nemotron35Lightning30BA3B.entry_function().params
    files = sorted(pathlib.Path(a.acts).glob("*.pt"))
    acts, j = {}, 0
    for p in params:
        if p.is_const:
            continue
        acts[p.name] = torch.load(files[j], map_location=dev)
        j += 1
    acts["cur_pos"] = int(acts["cur_pos"][0])

    layer = a.layer if a.layer is not None else model.LAYER_KINDS.index("moe")
    if model.LAYER_KINDS[layer] != "moe":
        raise SystemExit(f"layer {layer} is {model.LAYER_KINDS[layer]}, not MoE")

    h = R._ops_step(twin._bound, acts, stop=layer).reshape(1, 1, H).to(BF16)
    w = twin._bound
    args = (h.reshape(H), w[f"l{layer}_gamma"].reshape(H),
            w[f"l{layer}_w_router"].reshape(E, H),
            w[f"l{layer}_e_bias"].reshape(E).float(),
            w[f"l{layer}_w_up"].reshape(E, I, H),
            w[f"l{layer}_w_down"].reshape(E, H, I),
            w[f"l{layer}_w_sh_up"].reshape(IS, H),
            w[f"l{layer}_w_sh_down"].reshape(H, IS))
    ref = reference(*args)

    from kernels import ops  # noqa: PLC0415
    ext = ops()
    call = lambda: ext.moe_layer(  # noqa: E731
        args[0], args[1], args[2].reshape(-1), args[3], args[4].reshape(-1),
        args[5].reshape(-1), args[6].reshape(-1), args[7].reshape(-1))
    got = call()
    mine = dict(zip(["h_out", "idx", "gw", "h2", "mid", "smid", "acc"], got))

    print(f"layer {layer}  |h| = {h.float().norm().item():.4e}")
    same_set = set(mine["idx"].tolist()) == set(ref["idx"].tolist())
    print(f"expert set: mine {sorted(mine['idx'].tolist())}  "
          f"ref {sorted(ref['idx'].tolist())}   {'same' if same_set else 'DIFFER'}")

    landings = {"h2": 2, "smid": 4, "acc": 5, "h_out": 12}
    bad = 0 if same_set else 1
    print(f"{'stage':8} {'rel_l2':>12}  {'bound':>10}  {'|ref|':>10}   verdict")
    for name in ("h2", "smid", "acc", "h_out"):
        bound = ULP * (landings[name] ** 0.5)
        r = rel_l2(mine[name], ref[name])
        live = ref[name].reshape(-1).float().abs().mean().item()
        ok = r <= bound and live > 1e-8
        bad += not ok
        print(f"{name:8} {r:12.3e}  {bound:10.3e}  {live:10.3e}   {'ok' if ok else 'FAIL'}")

    if a.bench:
        import time  # noqa: PLC0415
        for _ in range(max(8, a.bench // 8)):
            call()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.bench):
            call()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / a.bench * 1e6
        print(f"\nmoe layer: {us:8.1f} us/layer   x23 = {us * 23 / 1e3:6.3f} ms/step")
    print("PASS" if not bad else f"{bad} check(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
