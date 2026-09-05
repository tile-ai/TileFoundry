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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-scale", type=float, default=1.0)
    a = ap.parse_args()

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
    print("PASS" if not bad else f"{bad} stage(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
