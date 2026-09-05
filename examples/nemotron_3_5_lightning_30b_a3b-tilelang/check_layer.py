"""Compare the handwritten Mamba-2 layer against the op-by-op reference, stage by stage.

`check_all.py` has one setting -- the whole step, 59 outputs -- and the mega
kernel has no per-layer entry, so neither can say which stage of a layer went
wrong. This runs the layer's five stages against the same torch expressions
`runtime_model._ops_step` uses and reports each one, so a disagreement names its
own stage instead of arriving as a wrong hidden row.

    python check_layer.py [--seed N] [--tol-scale F]
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import model

H, MH, MD, SS, NG, MI, CONV, KER, WIN, PROJ, GRP, HPG = (
    model.H, model.MH, model.MD, model.SS, model.NG, model.MI, model.CONV,
    model.KER, model.WIN, model.PROJ, model.GRP, model.HPG)
EPS, DTMIN = model.EPS, model.DTMIN
BF16 = torch.bfloat16

#: One bf16 round-to-nearest is at most 2^-9 relative; a stage k roundings deep
#: is bounded by the random-walk sum, which is what `check_all.py` argues too.
ULP = 2.0 ** -9


def reference(h, gamma, w_in, w_out, conv_w, conv_b, ggdn, mscal, conv_state, ssm_state):
    """The Mamba-2 branch of `runtime_model._ops_step`, operation for operation."""
    a_log, dt_bias, d_skip = mscal[0], mscal[1], mscal[2]
    hh = h.reshape(1, 1, H)
    xf = hh.float()
    nz = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(BF16)
    h2 = (nz * gamma.reshape(1, 1, -1)).reshape(1, H)

    gate = h2 @ w_in[0:MI].t()
    col0 = h2 @ w_in[MI:MI + CONV].t()
    dt = h2 @ w_in[MI + CONV:PROJ].t()
    proj = torch.cat([gate.reshape(-1), col0.reshape(-1), dt.reshape(-1)])

    win = torch.cat([conv_state.reshape(1, CONV, WIN), col0.reshape(1, CONV, 1)], dim=2)
    conv_out = win[:, :, 1:KER].contiguous()
    cs = (win * conv_w.reshape(1, CONV, KER)).sum(-1)
    xbc = F.silu(cs + conv_b.reshape(1, CONV))

    x = xbc[:, 0:MI].reshape(1, MH, MD)
    bg = xbc[:, MI:MI + NG * SS].reshape(1, NG, SS)
    cg = xbc[:, MI + NG * SS:CONV].reshape(1, NG, SS)
    b = bg.repeat_interleave(HPG, dim=1).reshape(1, MH, 1, SS)
    c = cg.repeat_interleave(HPG, dim=1).reshape(1, MH, SS, 1)
    dta = F.softplus(dt.reshape(1, MH, 1) + dt_bias.reshape(1, MH, 1)).clamp(min=DTMIN)
    dte = dta.reshape(1, MH, 1, 1)
    an = (-torch.exp(a_log.float())).reshape(1, MH, 1, 1)
    da = torch.exp(dte.float() * an)
    dbx = ((dte * b) * x.reshape(1, MH, MD, 1)).float()
    ssm_out = ssm_state.reshape(1, MH, MD, SS) * da + dbx
    y = torch.matmul(ssm_out.to(BF16), c.to(BF16)).reshape(1, MH, MD)

    yd = y + x * d_skip.reshape(1, MH, 1)
    yf = yd.reshape(1, MI).float() * F.silu(gate.float())
    yg = yf.reshape(1, NG, GRP)
    yn = (yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + EPS)).reshape(1, MI)
    scan = ggdn.reshape(1, MI) * yn.to(BF16)
    mix = (scan @ w_out.t()).reshape(1, 1, H)
    return {
        "proj": proj, "xbc": xbc.reshape(-1), "y": y.reshape(-1),
        "scan": scan.reshape(-1), "h_out": (hh + mix).reshape(-1),
        "conv_out": conv_out.reshape(-1), "ssm_out": ssm_out.reshape(-1),
    }


def rel_l2(a, b):
    a, b = a.float(), b.float()
    denom = b.norm().item()
    return (a - b).norm().item() / (denom if denom else 1.0)


def _real(a) -> int:
    """One layer of the checkpoint, on a hidden row a real step produced.

    A synthetic draw does not reproduce the checkpoint's weight distribution or
    the magnitude the residual stream reaches by the middle of the model, and a
    stage whose error only shows up there is invisible without this.
    """
    import pathlib as _pl  # noqa: PLC0415

    import paths  # noqa: PLC0415
    import runtime_model as R  # noqa: PLC0415
    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415

    dev = "cuda:0"
    twin = R.Nemotron35Lightning30BA3BRuntime()
    R.set_impl("ops")
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=dev))

    params = model.Nemotron35Lightning30BA3B.entry_function().params
    files = sorted(_pl.Path(a.acts).glob("*.pt"))
    if not files:
        raise SystemExit(f"no *.pt under {a.acts}; run dump_acts.py first")
    acts, j = {}, 0
    for p in params:
        if p.is_const:
            continue
        acts[p.name] = torch.load(files[j], map_location=dev)
        j += 1
    acts["cur_pos"] = int(acts["cur_pos"][0])

    layer = a.layer
    if layer is None:
        layer = model.LAYER_KINDS.index("linear_attention")
    if model.LAYER_KINDS[layer] != "linear_attention":
        raise SystemExit(f"layer {layer} is {model.LAYER_KINDS[layer]}, not Mamba-2")

    h = R._ops_step(twin._bound, acts, stop=layer).reshape(1, 1, H).to(BF16)
    w = twin._bound
    args = (h.reshape(H), w[f"l{layer}_gamma"].reshape(H),
            w[f"l{layer}_w_in"].reshape(PROJ, H), w[f"l{layer}_w_out"].reshape(H, MI),
            w[f"l{layer}_conv_w"].reshape(CONV, KER), w[f"l{layer}_conv_b"].reshape(CONV),
            w[f"l{layer}_gamma_gdn"].reshape(MI),
            torch.stack([w[f"l{layer}_a_log"].reshape(MH),
                         w[f"l{layer}_dt_bias"].reshape(MH),
                         w[f"l{layer}_d_skip"].reshape(MH)]),
            acts[f"l{layer}_conv_state"].reshape(CONV, WIN),
            acts[f"l{layer}_ssm_state"].reshape(MH, MD, SS).float())
    print(f"layer {layer}  |h| = {h.float().norm().item():.4e}")
    return _compare(a, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-scale", type=float, default=1.0)
    ap.add_argument("--real", action="store_true",
                    help="use the checkpoint's weights and a real hidden row, "
                         "not a synthetic draw")
    ap.add_argument("--layer", type=int, default=None,
                    help="with --real: which Mamba layer (default: the first)")
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--bench", type=int, default=0,
                    help="also time the layer over this many iterations")
    ap.add_argument("--prepared", default=None)
    a = ap.parse_args()

    if a.real:
        return _real(a)

    torch.manual_seed(a.seed)
    dev = "cuda"

    def rb(*shape, scale=1.0):
        return (torch.randn(*shape, device=dev) * scale).to(BF16)

    #: Weight magnitudes near the checkpoint's, so the bf16 error the bounds
    #: below argue about is the error the real thing makes.
    h = rb(H, scale=0.5)
    gamma = rb(H, scale=1.0)
    w_in = rb(PROJ, H, scale=0.02)
    w_out = rb(H, MI, scale=0.02)
    conv_w = rb(CONV, KER, scale=0.3)
    conv_b = rb(CONV, scale=0.1)
    ggdn = rb(MI, scale=1.0)
    conv_state = rb(CONV, WIN, scale=0.5)
    ssm_state = torch.randn(MH, MD, SS, device=dev) * 0.3

    #: The three scales are drawn the way Mamba-2 initialises them, not from a
    #: unit normal. `a_log` is the log of a state decay in [1, 16] and `dt_bias`
    #: is the inverse-softplus of a step in [0.001, 0.1], so both land where the
    #: exponentials are worked hardest. A unit-normal draw keeps every argument
    #: near zero, where an approximate `expf` agrees with an accurate one and a
    #: precision fault this test exists to catch does not show up at all.
    decay = torch.rand(MH, device=dev) * (16.0 - 1.0) + 1.0
    a_log = torch.log(decay)
    step = torch.rand(MH, device=dev) * (0.1 - 0.001) + 0.001
    dt_bias = step + torch.log(-torch.expm1(-step))
    d_skip = torch.randn(MH, device=dev) * 0.5 + 1.0
    mscal = torch.stack([a_log.to(BF16), dt_bias.to(BF16), d_skip.to(BF16)])

    return _compare(a, h, gamma, w_in, w_out, conv_w, conv_b, ggdn, mscal,
                    conv_state, ssm_state)


def _compare(a, h, gamma, w_in, w_out, conv_w, conv_b, ggdn, mscal,
             conv_state, ssm_state) -> int:
    """Run both sides over one layer's inputs and report every stage."""
    ref = reference(h, gamma, w_in, w_out, conv_w, conv_b, ggdn, mscal,
                    conv_state, ssm_state)

    from kernels import ops  # noqa: PLC0415
    got = ops().mamba_layer(h, gamma, w_in.reshape(-1), w_out.reshape(-1),
                            conv_w.reshape(-1), conv_b, ggdn, mscal.reshape(-1),
                            conv_state.reshape(-1), ssm_state.reshape(-1))
    names = ["h_out", "conv_out", "ssm_out", "proj", "xbc", "y", "scan"]
    mine = dict(zip(names, got))

    #: bf16 landings on the way to each stage, counted off the reference above.
    landings = {"proj": 3, "xbc": 5, "conv_out": 3, "y": 9, "ssm_out": 8,
                "scan": 12, "h_out": 14}
    order = ["proj", "conv_out", "xbc", "ssm_out", "y", "scan", "h_out"]

    print(f"{'stage':10} {'rel_l2':>12}  {'bound':>10}  {'|ref|':>10}   verdict")
    bad = 0
    for name in order:
        k = landings[name]
        bound = ULP * (k ** 0.5) * a.tol_scale
        r = rel_l2(mine[name].reshape(-1), ref[name].reshape(-1))
        got_t = mine[name].reshape(-1).float()
        ref_t = ref[name].reshape(-1).float()
        # A stage whose reference is all zeros would pass any bound, so the
        # magnitude is a criterion too: this reports what it compared.
        live = ref_t.abs().mean().item()
        alive = live > 1e-6 and (ref_t != 0).float().mean().item() > 0.5
        ok = r <= bound and torch.isfinite(got_t).all().item() and alive
        bad += not ok
        note = "ok" if ok else ("FAIL (reference is degenerate)" if not alive else "FAIL")
        print(f"{name:10} {r:12.3e}  {bound:10.3e}  {live:10.3e}   {note}")
    if a.bench:
        _bench(a.bench, ops(), mine, got, h, gamma, w_in, w_out, conv_w, conv_b,
               ggdn, mscal, conv_state, ssm_state)
    print("PASS" if not bad else f"{bad} stage(s) FAILED")
    return 1 if bad else 0


def _bench(iters, ext, mine, got, h, gamma, w_in, w_out, conv_w, conv_b, ggdn,
           mscal, conv_state, ssm_state) -> None:
    """Time one layer, warmed, on the stream the launches actually use."""
    import time  # noqa: PLC0415

    call = lambda: ext.mamba_layer(  # noqa: E731
        h.reshape(H), gamma.reshape(H), w_in.reshape(-1), w_out.reshape(-1),
        conv_w.reshape(-1), conv_b.reshape(CONV), ggdn.reshape(MI),
        mscal.reshape(-1), conv_state.reshape(-1), ssm_state.reshape(-1))
    for _ in range(max(8, iters // 8)):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    # 23 of the 52 layers are this one, so the step-level cost it implies is
    # what a whole-step number has to be read against.
    print(f"\nmamba layer: {us:8.1f} us/layer   x23 = {us * 23 / 1e3:6.3f} ms/step")


if __name__ == "__main__":
    raise SystemExit(main())
