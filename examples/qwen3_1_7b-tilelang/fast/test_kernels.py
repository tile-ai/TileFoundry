"""Each kernel against a torch statement of the same thing, at real dimensions."""
from __future__ import annotations

import math
import sys

import torch

import kernels as K

DEV = "cuda:0"
H, HQ, HKV, D, I, V = 2048, 16, 8, 128, 6144, 151936
EPS = 1e-6
MP = 4096
FAIL = []


def report(name, got, ref, rtol=2e-2, atol=2e-2):
    got32, ref32 = got.float(), ref.float()
    err = (got32 - ref32).abs()
    scale = ref32.abs().max().clamp(min=1e-6)
    rel = (err.max() / scale).item()
    cos = torch.nn.functional.cosine_similarity(
        got32.flatten(), ref32.flatten(), dim=0
    ).item()
    ok = rel < rtol and cos > 1 - atol
    print(f"{'ok  ' if ok else 'FAIL'} {name:22s} max_rel={rel:.3g} cosine={cos:.6f}")
    if not ok:
        FAIL.append(name)


def t_rms_norm():
    x = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(H, device=DEV, dtype=torch.bfloat16)
    K.rms_norm(H, EPS)(x, g, out)
    x32 = x.float()
    ref = (x32 * torch.rsqrt(x32.pow(2).mean() + EPS)).to(torch.bfloat16) * g
    report("rms_norm", out, ref)


def t_resid_rms_norm():
    SK = 8
    a = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    p = torch.randn(SK, H, device=DEV, dtype=torch.float32)
    g = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    ho = torch.empty(H, device=DEV, dtype=torch.bfloat16)
    xn = torch.empty(H, device=DEV, dtype=torch.bfloat16)
    K.resid_rms_norm(H, SK, EPS)(a, p, g, ho, xn)
    hr = a + p.sum(0).to(torch.bfloat16)
    report("resid_rms_norm.h", ho, hr)
    h32 = hr.float()
    ref = (h32 * torch.rsqrt(h32.pow(2).mean() + EPS)).to(torch.bfloat16) * g
    report("resid_rms_norm.xn", xn, ref)


def t_gemv():
    for Kd, N, BN, BK, SK, thr in [
        (H, 4096, 256, 64, 8, 256), (H, H, 128, 128, 8, 128),
        (H, 2 * I, 256, 64, 2, 256), (I, H, 128, 128, 8, 128),
    ]:
        x = torch.randn(Kd, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(Kd, N, device=DEV, dtype=torch.bfloat16) / Kd**0.5
        p = torch.empty(SK, N, device=DEV, dtype=torch.float32)
        K.gemv(Kd, N, BN, BK, SK, thr)(x, w, p)
        report(f"gemv {Kd}x{N}", p.sum(0), x.float() @ w.float(), rtol=5e-3)


def t_lm_head():
    x = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(H, V, device=DEV, dtype=torch.bfloat16) / H**0.5
    kern, NB = K.lm_head(H, V, 128, 128, 128)
    o = torch.empty(V, device=DEV, dtype=torch.float32)
    bv = torch.empty(NB, device=DEV, dtype=torch.float32)
    bi = torch.empty(NB, device=DEV, dtype=torch.int32)
    kern(x, w, o, bv, bi)
    ref = x.float() @ w.float()
    report("lm_head.logits", o, ref, rtol=5e-3)
    got_idx = int(bi[int(bv.argmax())])
    exp_idx = int(ref.argmax())
    ok = got_idx == exp_idx
    print(f"{'ok  ' if ok else 'FAIL'} lm_head.argmax        got={got_idx} want={exp_idx}")
    if not ok:
        FAIL.append("lm_head.argmax")


def _rope_ref(x, cos, sin, pos):
    """x (heads, D) bf16 -> rotated, matching HF apply_rotary_pos_emb."""
    c, s = cos[pos].float(), sin[pos].float()
    x1, x2 = x.float()[:, : D // 2], x.float()[:, D // 2:]
    half = torch.cat([-x2, x1], dim=-1)
    return (x.float() * c + half * s).to(torch.bfloat16)


def t_qk_rope_cache():
    SK, CAP, pos = 8, 512, 37
    scale = D**-0.5
    p = torch.randn(SK, HQ * D + 2 * HKV * D, device=DEV, dtype=torch.float32) / SK
    gq = torch.randn(D, device=DEV, dtype=torch.bfloat16)
    gk = torch.randn(D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(MP, D, device=DEV, dtype=torch.bfloat16)
    sin = torch.randn(MP, D, device=DEV, dtype=torch.bfloat16)
    pt = torch.tensor([pos], device=DEV, dtype=torch.int32)
    kc = torch.zeros(CAP, HKV * D, device=DEV, dtype=torch.bfloat16)
    vc = torch.zeros(CAP, HKV * D, device=DEV, dtype=torch.bfloat16)
    q = torch.empty(HQ * D, device=DEV, dtype=torch.bfloat16)
    K.qk_rope_cache(HQ, HKV, D, MP, CAP, SK, EPS)(p, gq, gk, cos, sin, pt, pt, kc, vc, q)

    flat = p.sum(0).to(torch.bfloat16).float()
    qr = flat[: HQ * D].view(HQ, D)
    kr = flat[HQ * D: HQ * D + HKV * D].view(HKV, D)
    vr = flat[HQ * D + HKV * D:].view(HKV, D)

    def nrm(t, g):
        return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS)).to(torch.bfloat16) * g

    report("qk_rope.q", q.view(HQ, D), _rope_ref(nrm(qr, gq), cos, sin, pos))
    report("qk_rope.k", kc[pos].view(HKV, D), _rope_ref(nrm(kr, gk), cos, sin, pos))
    report("qk_rope.v", vc[pos].view(HKV, D), vr.to(torch.bfloat16))


def _attn_ref(q, kc, vc, cur):
    """q (HQ,D) bf16 scaled; kc/vc (cur,HKV,D) -> (HQ*D,) bf16, f32 softmax."""
    G = HQ // HKV
    k = kc[:cur].float().repeat_interleave(G, dim=1)     # (cur, HQ, D)
    v = vc[:cur].float().repeat_interleave(G, dim=1)
    s = (q.float().unsqueeze(0) * k).sum(-1) * D**-0.5   # (cur, HQ)
    w = torch.softmax(s, dim=0)
    return (w.unsqueeze(-1) * v).sum(0).to(torch.bfloat16).reshape(-1)


def t_attn():
    CAP, SS, BS = 2560, 256, 128
    for cur in (1, 5, 256, 257, 700, 2048):
        pos = cur - 1
        q = torch.randn(HQ, D, device=DEV, dtype=torch.bfloat16)
        kc = torch.zeros(CAP, HKV, D, device=DEV, dtype=torch.bfloat16)
        vc = torch.zeros(CAP, HKV, D, device=DEV, dtype=torch.bfloat16)
        kc[:cur].normal_()
        vc[:cur].normal_()
        pt = torch.tensor([pos], device=DEV, dtype=torch.int32)
        NS = CAP // SS
        op = torch.empty(NS, HQ, D, device=DEV, dtype=torch.float32)
        mp = torch.empty(NS, HQ, device=DEV, dtype=torch.float32)
        lp = torch.empty(NS, HQ, device=DEV, dtype=torch.float32)
        out = torch.empty(HQ * D, device=DEV, dtype=torch.bfloat16)
        K.attn_partial(HQ, HKV, D, CAP, SS, D**-0.5)(
            q.reshape(-1), kc.view(CAP, -1), vc.view(CAP, -1), pt, op, mp, lp)
        K.attn_combine(HQ, D, CAP, SS)(op, mp, lp, out)
        report(f"attn cur={cur}", out, _attn_ref(q, kc, vc, cur), rtol=3e-2, atol=3e-2)


def t_silu_mul():
    SK = 2
    p = torch.randn(SK, 2 * I, device=DEV, dtype=torch.float32)
    o = torch.empty(I, device=DEV, dtype=torch.bfloat16)
    K.silu_mul(I, SK)(p, o)
    g = p.sum(0)[:I].to(torch.bfloat16)
    u = p.sum(0)[I:].to(torch.bfloat16)
    report("silu_mul", o, torch.nn.functional.silu(g.float()).to(torch.bfloat16) * u)


def t_embed():
    tbl = torch.randn(1024, H, device=DEV, dtype=torch.bfloat16)
    ids = torch.tensor([517], device=DEV, dtype=torch.int64)
    o = torch.empty(H, device=DEV, dtype=torch.bfloat16)
    K.embed(1024, H)(tbl, ids, o)
    report("embed", o, tbl[517])


def t_sample_step():
    NB, NSTEPS, PL = 1187, 64, 5
    bv = torch.randn(NB, device=DEV, dtype=torch.float32)
    bi = torch.randint(0, V, (NB,), device=DEV, dtype=torch.int32)
    inp = torch.arange(NSTEPS, device=DEV, dtype=torch.int32) + 100
    pl = torch.tensor([PL], device=DEV, dtype=torch.int32)
    sam = torch.zeros(NSTEPS, device=DEV, dtype=torch.int32)
    kern = K.sample_step(NB, NSTEPS)
    for pos, expect_prompt in [(2, True), (PL - 1, False), (20, False)]:
        ids = torch.zeros(1, device=DEV, dtype=torch.int64)
        pt = torch.tensor([pos], device=DEV, dtype=torch.int32)
        kern(bv, bi, inp, pl, ids, pt, sam)
        best = int(bi[int(bv.argmax())])
        want_next = int(inp[pos + 1]) if expect_prompt else best
        ok = int(sam[pos]) == best and int(ids[0]) == want_next and int(pt[0]) == pos + 1
        print(f"{'ok  ' if ok else 'FAIL'} sample_step pos={pos:<3d}  "
              f"sam={int(sam[pos])} want={best} next={int(ids[0])} want={want_next}")
        if not ok:
            FAIL.append(f"sample_step:{pos}")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in [t_rms_norm, t_resid_rms_norm, t_gemv, t_lm_head,
               t_qk_rope_cache, t_attn, t_silu_mul, t_embed, t_sample_step]:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=6)
            FAIL.append(fn.__name__)
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
