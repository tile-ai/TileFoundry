#!/usr/bin/env python3
"""Scheduled HIR + finalized TIR of the attention kernels, per prompt length.

    scheduled/attention/attn_prep_T<T>.py -> tir/attention/attn_prep_T<T>.py
    scheduled/attention/attn_core_T<T>.py -> tir/attention/attn_core_T<T>.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: the `tilefoundry` CLI: $TILEFOUNDRY, or whatever is on PATH (this example is not
#: installed and does not carry a virtualenv of its own).
TILEFOUNDRY = os.environ.get("TILEFOUNDRY") or shutil.which("tilefoundry") or "tilefoundry"
sys.path.insert(0, str(ROOT))

PREP = '''"""Scheduled HIR: attention prep at prompt length {T}.

Implements model.py Qwen3_8AttentionLayer.gated_attention up to the core:
  q = partial_rope(rms_norm(qg[..., :256], 1 + gamma_q)), k = partial_rope_kv(rms_norm(k, 1 + gamma_k)), v.
Runtime kernel: runtime/kernels/attention.py make_prep(T) (`attn_prep`).

Decomposition (as the kernel): one CTA per token, 1024 threads = 32 warps; warp w < 24
handles query head w, 24 <= w < 28 key head w - 24, w >= 28 copies value head w - 28.
Each lane holds elements 4l..4l+3 and 128+4l..128+4l+3 of its head row; the RMS
statistic is one cross-lane reduce; the row is landed once in bf16.
RoPE (rotate-half on dims 0..63, bf16 arithmetic): the pair (d, d+32) must meet in one
lane. TileFoundry's H200 instruction set has no intra-warp shuffle (schedule facts:
copy, copy_async, copy_async_tensor, ldmatrix, tiled_mma, binary/unary/cast/where/
clamp/relu/reduce), so this program states the exchange through shared memory: the
normalized row is staged in smem and lane j reads the pair (j, j+32). The kernel
realizes the same pairing in registers: the pair sits in lanes l and l^8 and is
exchanged with shfl_xor(., 8). Values and roundings are identical (tested bitwise).
Result: scheduled lowering accepts one tensor result, so q, k and v are stated as one
(T, 32, 256) tensor whose head slots 0..23 / 24..27 / 28..31 are q / k / v -- the
kernel's warp-slot order; the kernel writes the same rows into three buffers.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = {T}
D = 256
ROT = 64
HALF = 32
EPS = 1e-6
QW = 24 * 512


@module(entry="attn_prep", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", T_), Topology("thread", 1024)))
class SCHED_ATTN_PREP_T{T}:
    @func
    def attn_prep(qkv: Tensor[(T_, 14336), "bf16"], gamma_q: Tensor[(D,), "bf16"], gamma_k: Tensor[(D,), "bf16"],
                  cos: Tensor[(T_, ROT), "bf16"], sin: Tensor[(T_, ROT), "bf16"]):
        with Mesh(("cta",), layout=(T_,), names=("t",)) as cta:
            with Mesh(("thread",), layout=(32, 32), names=("warp", "lane")) as th:
                out = tf.zeros(Tensor[(T_, 32, D), "bf16"])
{BODY}
            return out
'''

ROLE = '''                with Mesh(th[{w0}:{w1}, :], layout=({H}, 32), names=("head", "lane")) as m{tag}:
                    x = tf.schedule((src{tag}, tf.empty(Tensor[({H}, D), "bf16", (({H} @ m{tag}.head, 2, 32 @ m{tag}.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    g = tf.schedule((tf.reshape({gamma}, new_shape=(1, D)), tf.empty(Tensor[(1, D), "bf16", ((1, 2, 32 @ m{tag}.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    xf = tf.cast(x, dtype="f32")
                    ms = tf.reduce(xf * xf, axes=(-1,), keepdim=True, kind="mean")
                    n = tf.cast(xf * tf.rsqrt(ms + EPS) * (tf.cast(g, dtype="f32") + 1.0), dtype="bf16")
                    ns = tf.schedule((n, tf.empty(Tensor[({H}, D), "bf16", ({H}, D), "smem"])), op=T.copy())
                    lo = tf.cast(tf.schedule((ns[:, 0:HALF], tf.empty(Tensor[({H}, HALF), "bf16", (({H} @ m{tag}.head, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    hi = tf.cast(tf.schedule((ns[:, HALF:ROT], tf.empty(Tensor[({H}, HALF), "bf16", (({H} @ m{tag}.head, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_lo = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_hi = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_lo = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_hi = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ m{tag}.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    r_lo = tf.cast(tf.cast(tf.cast(lo * c_lo, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(tf.neg(hi) * s_lo, dtype="bf16"), dtype="f32"), dtype="bf16")
                    r_hi = tf.cast(tf.cast(tf.cast(hi * c_hi, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(lo * s_hi, dtype="bf16"), dtype="f32"), dtype="bf16")
                    o_lo = tf.schedule((tf.reshape(r_lo, new_shape=(1, {H}, HALF)), tf.empty(Tensor[(1, {H}, HALF), "bf16", (1, {H}, HALF)])), op=T.copy())
                    o_hi = tf.schedule((tf.reshape(r_hi, new_shape=(1, {H}, HALF)), tf.empty(Tensor[(1, {H}, HALF), "bf16", (1, {H}, HALF)])), op=T.copy())
                    o_pass = tf.schedule((tf.reshape(ns[:, ROT:D], new_shape=(1, {H}, D - ROT)), tf.empty(Tensor[(1, {H}, D - ROT), "bf16", (1, {H}, D - ROT)])), op=T.copy())
                    out = tf.insert_slice(out, o_lo, (cta.t, {w0}, 0))
                    out = tf.insert_slice(out, o_hi, (cta.t, {w0}, HALF))
                    out = tf.insert_slice(out, o_pass, (cta.t, {w0}, ROT))
'''

VCOPY = '''                with Mesh(th[28:32, :], layout=(4, 32), names=("head", "lane")) as mv:
                    vv = tf.schedule((tf.reshape(qkv[cta.t:cta.t + 1, QW + 1024:QW + 2048], new_shape=(4, D)), tf.empty(Tensor[(4, D), "bf16", ((4 @ mv.head, 32 @ mv.lane, 8), (0, 0, 1)), "rmem"])), op=T.copy())
                    vo = tf.schedule((tf.reshape(vv, new_shape=(1, 4, D)), tf.empty(Tensor[(1, 4, D), "bf16", (1, 4, D)])), op=T.copy())
                    out = tf.insert_slice(out, vo, (cta.t, 28, 0))
'''


def prep_source(T):
    srcq = "                srcq = tf.reshape(qkv[cta.t:cta.t + 1, 0:QW], new_shape=(24, 512))[:, 0:D]\n"
    srck = "                srck = tf.reshape(qkv[cta.t:cta.t + 1, QW:QW + 1024], new_shape=(4, D))\n"
    body = (srcq + ROLE.format(w0=0, w1=24, H=24, tag="q", gamma="gamma_q", out="q_out")
            + srck + ROLE.format(w0=24, w1=28, H=4, tag="k", gamma="gamma_k", out="k_out") + VCOPY)
    return PREP.format(T=T, BODY=body.rstrip("\n"))


CORE = '''"""Scheduled HIR: attention core at prompt length {T} (heads per CTA {HPC}, gate "{MODE}", DSPLIT {DSPLIT}).

Implements model.py Qwen3_8AttentionLayer.gated_attention from the core on:
  y = bf16(bf16(attend(q, k, v)) * bf16(sigmoid(gate))) for the 24 query heads, causal, GQA 6:1.
Runtime kernel: runtime/kernels/attention.py make_core({T}, {HPC}, "{MODE}", DSPLIT={DSPLIT}) (`attn_core`),
configuration from attention.CORE_CONFIG[{T}].

Decomposition (as the kernel): CTA mesh (token block, query-head group{DS_NAME}), the head
axis fastest and the token block walked from the last (longest causal range) down
(LPT order): bx = NB - 1 - blk. Warpgroup 0 is the producer (warp 0: Q tiles and the
K ring, warp 1: the V ring, TMA into 128B-swizzled tiles, ST = 2 deep); warpgroup
1 + w computes query head {HEAD_EXPR} + w with wgmma: S = Q K^T (SS, n = 64, K-major),
online softmax in base 2 (scale 1/16 folded in c = log2(e)/16), O = O * alpha + bf16(P) V
(RS, n = {DV}, V MN-major), P landed in bf16 from the S accumulator registers.
The key blocks below the diagonal run in the loop; the diagonal block (the only one that
crosses it, TM = BN = 64) is peeled after the loop and gets the causal mask. The mask
tile is a constant input `diag` (0 on/below the diagonal, -inf above); the kernel
evaluates the same predicate from indices (T.if_then_else(..., S, -inf) == S + diag).
{GATE_DOC}
Rounding (HIR attend): f32 scores, running max and sum in f32, P bf16 for the PV wgmma,
f32 accumulation, O / l landed in bf16, gate product in f32 landed in bf16.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = {T}
D = 256
DV = {DV}
TM = 64
BN = 64
ST = 2
NB = T_ // TM
NQ = 24
NG = {NG}
LOG2E = 1.4426950408889634
NLOG2E = -1.4426950408889634
C = LOG2E / 16.0
NEG = float("-inf")
QKV = 14336


@module(entry="attn_core", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NG * NB{DS_MUL}), Topology("thread", {THREADS})))
class SCHED_ATTN_CORE_T{T}:
    @func
    def attn_core(q: Tensor[(T_, NQ, D), "bf16"], k: Tensor[(T_, 4, D), "bf16"], v: Tensor[(T_, 4, D), "bf16"],
                  qkv: Tensor[(T_, QKV), "bf16"], diag: Tensor[(TM, BN), "f32"]) -> Tensor[(T_, NQ * D), "bf16"]:
        with Mesh(("cta",), layout=(NB, NG{DS_LAYOUT}), names=("blk", "hg"{DS_NAMES})) as cta:
            with Mesh(("thread",), layout=({WGS}, 128), names=("warpgroup", "participant")) as threads:
                s_mma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                o_mma = T.cuda.sm90.Wgmma(n=DV, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(T_, NQ * D), "bf16"])
                bx = NB - 1 - cta.blk
                g = cta.hg // {GPK}
{BODY}
                return out
'''

LAY_KM = "Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)"     # (64, 256) K-major tile
LAY_KT = "Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3)"     # its transpose (wgmma B view)


def lay_v(dv):
    """(64, dv) V tile, MN-major for the PV wgmma: 16-row k groups of dv/64 128B panels."""
    return f"Layout(((4, 2, 8), ({dv // 64}, 64)), (({dv * 16}, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)"


def frag(n, c):
    """wgmma accumulator fragment of a (64, n) tile on one warpgroup."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, 512, 128, 64))"


def pfrag(c):
    """The (64, 64) P tile as the RS A operand: the 4 k16 tiles lead."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, 4, 2, 4 @ {c}.lane4, 2), (16, 8, 1, 1024, 512, 128, 64))"


def core_source(T):
    from runtime.kernels.attention import CORE_CONFIG
    hpc, mode, ds = CORE_CONFIG[T]
    dv = 256 // ds
    early = mode == "early"
    nb = T // 64
    sigw = 4 if nb <= 2 else 2
    i8, i12, i16, i20, i24 = (" " * n for n in (8, 12, 16, 20, 24))
    head = (lambda w: f"cta.hg * {hpc} + {w}") if hpc == 2 else (lambda w: "cta.hg")
    dcol = " + cta.dh * DV" if ds == 2 else ""
    L = []

    def consumer(ind, w, rng, diagonal):
        c = f"_c{w}"
        kblk = "bx" if diagonal else "j"
        L.append(f"{ind}with Mesh(threads[{w + 1}:{w + 2}, :], layout=(4, 8, 4), names=(\"warp\", \"lane8\", \"lane4\")) as {c}:")
        b = ind + "    "
        L.append(f"{b}S{w} = tf.schedule((tf.zeros(Tensor[(TM, BN), \"f32\", {frag(64, c)}, \"rmem\"]), qt{w}, tf.bitcast(kt, {LAY_KT})), op=T.tiled_mma(atom=s_mma), repeat=(1, 1, 16))")
        if diagonal:
            L.append(f"{b}S{w} = S{w} + dmask{w}")
        L.append(f"{b}m_new{w} = tf.max(m{w}, tf.reduce(S{w}, axes=(-1,), keepdim=True, kind=\"max\"))")
        L.append(f"{b}alpha{w} = tf.exp2((m{w} - m_new{w}) * C)")
        L.append(f"{b}P{w} = tf.exp2(S{w} * C - m_new{w} * C)")
        L.append(f"{b}l{w} = l{w} * alpha{w} + tf.reduce(P{w}, axes=(-1,), keepdim=True, kind=\"sum\")")
        L.append(f"{b}Pb{w} = tf.schedule((tf.cast(P{w}, dtype=\"bf16\"), tf.empty(Tensor[(TM, BN), \"bf16\", {pfrag(c)}, \"rmem\"])), op=T.copy())")
        L.append(f"{b}O{w} = tf.schedule((O{w} * alpha{w}, Pb{w}, vt), op=T.tiled_mma(atom=o_mma), repeat=(1, 1, 4))")
        L.append(f"{b}m{w} = m_new{w}")

    def loads(ind, kblk):
        L.append(f"{ind}with threads[0, :32] as _kl:")
        L.append(f"{ind}    k_ring = tf.empty(Tensor[(BN, D), \"bf16\", {LAY_KM}, \"smem\"], buffers=ST)")
        L.append(f"{ind}    kt = tf.schedule((tf.reshape(k[{kblk} * BN:{kblk} * BN + BN, g:g + 1, :], new_shape=(BN, D)), k_ring), op=T.copy_async_tensor())")
        L.append(f"{ind}with threads[0, 32:64] as _vl:")
        L.append(f"{ind}    v_ring = tf.empty(Tensor[(BN, DV), \"bf16\", {lay_v(dv)}, \"smem\"], buffers=ST)")
        vcol = "cta.dh * DV:cta.dh * DV + DV" if ds == 2 else ":"
        L.append(f"{ind}    vt = tf.schedule((tf.reshape(v[{kblk} * BN:{kblk} * BN + BN, g:g + 1, {vcol}], new_shape=(BN, DV)), v_ring), op=T.copy_async_tensor())")

    # Q tiles (+ the early gate) by the producer, consumer state init
    L.append(f"{i16}with threads[0, :32] as _ql:")
    for w in range(hpc):
        L.append(f"{i20}q_dst{w} = tf.empty(Tensor[(TM, D), \"bf16\", {LAY_KM}, \"smem\"])")
        L.append(f"{i20}qt{w} = tf.schedule((tf.reshape(q[bx * TM:bx * TM + TM, {head(w)}:{head(w)} + 1, :], new_shape=(TM, D)), q_dst{w}), op=T.copy_async_tensor())")
    gcol = lambda w: f"({head(w)}) * 512 + D{dcol}"
    if early:
        L.append(f"{i16}with threads[0, 32:64] as _gl:")
        L.append(f"{i20}g_dst = tf.empty(Tensor[(TM, DV), \"bf16\", (TM, DV), \"smem\"])")
        L.append(f"{i20}gt = tf.schedule((qkv[bx * TM:bx * TM + TM, {gcol(0)}:{gcol(0)} + DV], g_dst), op=T.copy_async_tensor())")
        if sigw == 4:
            L.append(f"{i16}with Mesh(threads[0, :], layout=(64, 2), names=(\"r\", \"h\")) as _s:")
            glay = "((64 @ _s.r, 2 @ _s.h, DV // 2), (0, 0, 1))"
        else:
            L.append(f"{i16}with Mesh(threads[0, 64:128], layout=(64,), names=(\"r\",)) as _s:")
            glay = "((64 @ _s.r, DV), (0, 1))"
        L.append(f"{i20}gr = tf.schedule((gt, tf.empty(Tensor[(TM, DV), \"bf16\", {glay}, \"rmem\"])), op=T.copy())")
        L.append(f"{i20}sig = tf.cast(1.0 / (tf.exp2(tf.cast(gr, dtype=\"f32\") * NLOG2E) + 1.0), dtype=\"bf16\")")
        L.append(f"{i20}gs = tf.schedule((sig, tf.empty(Tensor[(TM, DV), \"bf16\", (TM, DV), \"smem\"])), op=T.copy())")
    for w in range(hpc):
        c = f"_c{w}"
        L.append(f"{i16}with Mesh(threads[{w + 1}:{w + 2}, :], layout=(4, 8, 4), names=(\"warp\", \"lane8\", \"lane4\")) as {c}:")
        L.append(f"{i20}O{w} = tf.zeros(Tensor[(TM, DV), \"f32\", {frag(dv, c)}, \"rmem\"])")
        L.append(f"{i20}S0_{w} = tf.zeros(Tensor[(TM, BN), \"f32\", {frag(64, c)}, \"rmem\"])")
        L.append(f"{i20}m{w} = tf.reduce(S0_{w}, axes=(-1,), keepdim=True, kind=\"max\") + NEG")
        L.append(f"{i20}l{w} = tf.reduce(S0_{w}, axes=(-1,), keepdim=True, kind=\"sum\")")
        L.append(f"{i20}dmask{w} = tf.schedule((diag, tf.empty(Tensor[(TM, BN), \"f32\", {frag(64, c)}, \"rmem\"])), op=T.copy())")
    # key blocks strictly below the diagonal
    L.append(f"{i16}for j in range(NB - 1 - cta.blk):")
    loads(i20, "j")
    for w in range(hpc):
        consumer(i20, w, None, False)
    # the diagonal block
    loads(i16, "bx")
    for w in range(hpc):
        consumer(i16, w, None, True)
    # gate + epilogue
    if not early:
        L.append(f"{i16}with threads[0, :32] as _gl:")
        for w in range(hpc):
            L.append(f"{i20}g_dst{w} = tf.empty(Tensor[(TM, D), \"bf16\", {LAY_KM}, \"smem\"])")
            L.append(f"{i20}gt{w} = tf.schedule((qkv[bx * TM:bx * TM + TM, {gcol(w)}:{gcol(w)} + D], g_dst{w}), op=T.copy_async_tensor())")
    for w in range(hpc):
        c = f"_c{w}"
        L.append(f"{i16}with Mesh(threads[{w + 1}:{w + 2}, :], layout=(4, 8, 4), names=(\"warp\", \"lane8\", \"lane4\")) as {c}:")
        L.append(f"{i20}ob{w} = tf.cast(O{w} * (1.0 / l{w}), dtype=\"bf16\")")
        src = "gs" if early else f"gt{w}"
        L.append(f"{i20}gr{w} = tf.schedule(({src}, tf.empty(Tensor[(TM, DV), \"bf16\", {frag(dv, c)}, \"rmem\"])), op=T.copy())")
        sig = f"gr{w}" if early else f"tf.cast(1.0 / (tf.exp2(tf.cast(gr{w}, dtype=\"f32\") * NLOG2E) + 1.0), dtype=\"bf16\")"
        L.append(f"{i20}y{w} = tf.cast(tf.cast(ob{w}, dtype=\"f32\") * tf.cast({sig}, dtype=\"f32\"), dtype=\"bf16\")")
        L.append(f"{i20}staged{w} = tf.schedule((y{w}, tf.empty(Tensor[(TM, DV), \"bf16\", (TM, DV), \"smem\"])), op=T.copy())")
    L.append(f"{i16}with threads[0, :32] as _st:")
    for w in range(hpc):
        L.append(f"{i20}yt{w} = tf.schedule((staged{w}, tf.empty(Tensor[(TM, DV), \"bf16\", (TM, DV)])), op=T.copy_async_tensor())")
        L.append(f"{i20}out = tf.insert_slice(out, yt{w}, (bx * TM, ({head(w)}) * D{dcol}))")

    if early:
        gate_doc = ("Gate \"early\": warp 1 TMA-loads the gate tile before the main loop and the producer warps\n"
                    f"({'all 4' if sigw == 4 else 'warps 2-3'}) turn it into bf16(sigmoid) in shared memory while the consumer runs; the\n"
                    "epilogue multiplies O / l by it and stages y for one TMA store. (The kernel computes\n"
                    "the sigmoid in place in the gate buffer; this program names a second buffer.)")
    else:
        gate_doc = ("Gate \"late\": after the last K tile, warp 0 TMA-loads the gate tile of each head (the\n"
                    "kernel reuses the freed K ring slots nkb + w; this program names separate buffers);\n"
                    "the consumer applies the sigmoid in its epilogue and stages y (the kernel stages it in\n"
                    "the Q tile) for one TMA store.")
    return CORE.format(T=T, HPC=hpc, MODE=mode, DSPLIT=ds, DV=dv, NG=24 // hpc, THREADS=128 * (hpc + 1), WGS=hpc + 1,
                       DS_NAME=", output-dim half" if ds == 2 else "", DS_MUL=" * 2" if ds == 2 else "",
                       DS_LAYOUT=", 2" if ds == 2 else "", DS_NAMES=', "dh"' if ds == 2 else "",
                       HEAD_EXPR="hg * " + str(hpc) if hpc == 2 else "hg", GPK=6 // hpc, GATE_DOC=gate_doc, BODY="\n".join(L))


def main():
    sdir, tdir = ROOT / "scheduled/attention", ROOT / "tir/attention"
    sdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    fails = 0
    for T in [int(a) for a in sys.argv[1:]] or (128, 256, 512, 1024):
        for name, src in ((f"attn_prep_T{T}", prep_source(T)), (f"attn_core_T{T}", core_source(T))):
            (sdir / f"{name}.py").write_text(src)
            r = subprocess.run([TILEFOUNDRY, "schedule", "finalize", str(sdir / f"{name}.py"),
                                str(tdir / f"{name}.py")], capture_output=True, text=True)
            fails += r.returncode != 0
            print(f"{name:20s} finalize {'OK' if r.returncode == 0 else 'FAIL'}"
                  + ("" if r.returncode == 0 else "\n    " + r.stderr.strip()[:700].replace("\n", "\n    ")), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
