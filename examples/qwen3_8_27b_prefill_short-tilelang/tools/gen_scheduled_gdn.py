#!/usr/bin/env python3
"""Scheduled HIR + finalized TIR of the GatedDeltaNet mixer kernels, per prompt length.

    scheduled/gdn/gdn_prep_T<T>.py  -> tir/gdn/gdn_prep_T<T>.py
    scheduled/gdn/gdn_chunk_T<T>.py -> tir/gdn/gdn_chunk_T<T>.py
    scheduled/gdn/gdn_scan_T<T>.py  -> tir/gdn/gdn_scan_T<T>.py   (plain scan, z=None)
    scheduled/gdn/gdn_norm_T<T>.py  -> tir/gdn/gdn_norm_T<T>.py   (gated RMSNorm)

Runtime kernels: runtime/kernels/gdn.py make_prep / make_chunk / make_scan(gated=False)
(GdnKernels, precise state path) and make_gated_norm (gated RMSNorm on the plain read-out).
Each generated file's header names the model.py function it implements and lists every
place where the scheduled program differs from the kernel.

usage: tools/gen_scheduled_gdn.py [T ...] [--only prep|chunk|scan|norm] [--out DIR]
"""
import argparse
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

from runtime.kernels.gdn import A_OFF, B_OFF, EPS_L2_BF16, _prep_config  # noqa: E402

CONV = 10240
HV = 48
D = 128
C = 64
PROJ_PAD = 16640      # row stride of the engine's in-projection buffer (engine.GDN_IN_PAD)

PRE = '''from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

'''


# --------------------------------------------------------------------------------------
# gdn_prep
# --------------------------------------------------------------------------------------
PREP = '''"""Scheduled HIR: GDN prep at prompt length {T} (R = {R} rows per warp, W = {W} warps).

Implements model.py Qwen3_8DeltaNetLayer.gated_delta_net up to the delta rule:
  causal_conv (depthwise, f32 accumulation in tap order, landed in bf16, silu),
  l2_normalise of the q and k heads (bf16 arithmetic), the gates
  g = -exp(a_log) * softplus(a + dt_bias) (f32) and beta = sigmoid(b) (bf16),
  and the conv window (the last 3 rows of `mixed`).
Runtime kernel: runtime/kernels/gdn.py make_prep({T}, {PROJ_PAD}) (`gdn_prep`), R, W = _prep_config({T}).

Decomposition (as the kernel): CTA mesh (row block rb, head column hd), hd fastest
(tilelang grid (80, NB)); a CTA owns BT = R * W = {BT} rows of one 128-channel column
of the conv channels (hd < 16: a q head, hd < 32: a k head, else a v head).
{TH} threads = W warps x 32 lanes; warp w owns rows w*R .. w*R + R - 1 of the block,
lane l owns channels 4l .. 4l + 3 (4 conv taps of its 4 channels in 16 registers).
conv: f32 products of bf16 values summed in tap order, landed in bf16; silu in f32
(x / (exp(-x) + 1)), landed in bf16. q/k CTAs: square landed in bf16, row sum over the
128 channels (4 in-thread + a cross-lane reduce) in f32, landed in bf16, + bf16(1e-6)
landed, rsqrt landed, x * rsqrt landed. v CTAs: lanes 0..BT-1 of warp 0 compute g and
beta of value head hd - 32 for the block's rows. CTAs of the last row block
(rb = NB - 1) also copy the window rows T-3..T-1 of their 128 channels.
The roles are stated as CTA sub-meshes (q/k columns 0..31, v columns 32..79, window
row block NB-1); the kernel branches on hd and rb inside one code path.

Differences to the kernel:
- Zero padding of the conv: the kernel loads each thread's R + 3 rows with predicated
  global loads (rows < 0 read as 0) straight into registers. TileFoundry has no
  predicated load into registers; an indexed copy (gather) with zero fill exists only
  into shared memory (T.copy_async(fill=0.0), cp.async zero-fill), so this program
  gathers the block's BT + 3 rows (row indices rb*BT-3 .. rb*BT+BT-1, negative ->
  zero) into a shared tile and reads the 4 tap windows from it. The index vector is a
  constant input `rowidx` ((NB * (BT + 3),) i64, rowidx[b*(BT+3) + i] = b*BT - 3 + i):
  tf.arange has no TIR lowering (refused "unknown HIR call Arange") and its start
  cannot be a mesh coordinate.
- silu: the kernel takes a fast path (ex2.approx / rcp.approx) for all 4R values of a
  lane and one branch per thread redoes x / (1 + expf(-x)) exactly for every value whose
  fast result is within 16 f32 ulps of a bf16 rounding midpoint or whose x < -16; its
  output is bitwise that of the exact formula stated here.
- Cross-lane row sum of the l2norm: the kernel sums its 4 channels, then a shfl_xor
  tree over the 32 lanes. TileFoundry has no shuffle instruction; the reduce is stated
  as one tf.reduce over the lane-split axis (same values up to f32 summation order).
- softplus: the kernel computes log1p(exp(x)) for x <= 20. TileFoundry's unary kinds
  have no log1p and tf.softplus has no TIR lowering ("unknown HIR call Softplus"), so
  this program states where(x > 20, x, log(exp(x) + 1)) (log(1 + e) vs log1p(e): f32
  rounding only). sigmoid is stated as 1 / (exp(-b) + 1), the kernel's formula
  (tf.sigmoid: "unknown HIR call Sigmoid").
- g/beta threads: the kernel uses lanes 0..BT-1 of warp 0 (`tx < BT`). A thread-mesh
  slice inside a CTA sub-mesh is refused ("replacement scope selects positions ...
  outside parent scope positions ...; both must be continuous and each replacement
  selection must be contained in its parent selection"), so the v CTAs open a second
  thread mesh of BT threads (layout (BT,)), which TileFoundry places on threads
  0..BT-1 -- the same threads.
- Scalars a_log[h], dt_bias[h] are loaded as (1,) register tensors (a (1, 1) load
  triggers an ISL diagnostic in finalize; the values are the same).
- Stores: the kernel stores q/k/v as 8-byte bf16 vectors straight from its packed
  registers (lane = 4 channels); here each row block goes through a register tile copy.
- Result: scheduled lowering accepts one tensor result. The six outputs are one f32
  tensor (T + 3, {PACK}): rows 0..T-1, columns 0..10239 = q | k | v in conv-channel
  order (column 128*hd + c is channel c of head column hd), column 10240 + h = g[:, h],
  column 10288 + h = beta[:, h]; rows T..T+2, columns 0..10239 = window^T (row T + j
  = mixed row T-3+j). The kernel writes q, k, v, beta, window as bf16 into six
  buffers (window transposed (10240, 3)) and g as f32; every bf16 value here is that
  bf16 value widened exactly (the f32 slot is storage only, not a rounding change).
"""
{PRE}T_ = {T}
R = {R}
W = {W}
BT = R * W
NB = T_ // BT
CONV = {CONV}
PROJ = {PROJ_PAD}
A_OFF = {A_OFF}
B_OFF = {B_OFF}
G_COL = CONV
BETA_COL = CONV + 48
PACK = CONV + 96
EPS_L2 = {EPS}


@module(entry="gdn_prep", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NB * 80), Topology("thread", W * 32)))
class SCHED_GDN_PREP_T{T}:
    @func
    def gdn_prep(proj: Tensor[(T_, PROJ), "bf16"], conv_w: Tensor[(CONV, 4), "bf16", Layout((CONV, 4), (4, 1))],
                 a_log: Tensor[(48,), "bf16"], dt_bias: Tensor[(48,), "bf16"],
                 rowidx: Tensor[(NB * (BT + 3),), "i64"]) -> Tensor[(T_ + 3, PACK), "f32"]:
        with Mesh(("cta",), layout=(NB, 80), names=("rb", "hd")) as cta:
            with Mesh(("thread",), layout=(W * 32,), names=("t",)) as init:
                out = tf.zeros(Tensor[(T_ + 3, PACK), "f32"])
{BODY}
            return out
'''

CONV_BODY = '''                c0 = ({cs}.hd + {H0}) * 128
                with Mesh(("thread",), layout=(W, 32), names=("warp", "lane")) as th:
                    rows = rowidx[{cs}.rb * (BT + 3):{cs}.rb * (BT + 3) + BT + 3]
                    xs = tf.schedule((proj[:, c0:c0 + 128], tf.empty(Tensor[(BT + 3, 128), "bf16", (BT + 3, 128), "smem"]), rows), op=T.copy_async(fill=0.0))
                    taps = tf.bitcast(conv_w, Layout((4, CONV), (1, 4)))
                    wr = tf.cast(tf.schedule((taps[:, c0:c0 + 128], tf.empty(Tensor[(4, 128), "bf16", ((4, 32 @ th.lane, 4), (1, 0, 4)), "rmem"])), op=T.copy()), dtype="f32")
{TAPS}
                    xc = tf.cast(tf.cast(acc, dtype="bf16"), dtype="f32")
                    sv = tf.cast(xc / (tf.exp(tf.neg(xc)) + 1.0), dtype="bf16")
'''

TAP = '''                    x{j} = tf.schedule((xs[{j}:{j} + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    {lhs} = {prev}tf.cast(x{j}, dtype="f32") * wr[{j}:{j} + 1, :]
'''

QK_BODY = '''                    sf = tf.cast(sv, dtype="f32")
                    sq = tf.cast(tf.cast(sf * sf, dtype="bf16"), dtype="f32")
                    ssum = tf.cast(tf.reduce(sq, axes=(-1,), keepdim=True, kind="sum"), dtype="bf16")
                    se = tf.cast(tf.cast(ssum, dtype="f32") + EPS_L2, dtype="bf16")
                    rr = tf.cast(tf.cast(tf.rsqrt(tf.cast(se, dtype="f32")), dtype="bf16"), dtype="f32")
                    y = tf.cast(tf.cast(sf * rr, dtype="bf16"), dtype="f32")
                    yo = tf.schedule((y, tf.empty(Tensor[(BT, 128), "f32", (BT, 128)])), op=T.copy())
                    out = tf.insert_slice(out, yo, ({cs}.rb * BT, c0))
'''

V_BODY = '''                    vo = tf.schedule((tf.cast(sv, dtype="f32"), tf.empty(Tensor[(BT, 128), "f32", (BT, 128)])), op=T.copy())
                    out = tf.insert_slice(out, vo, ({cs}.rb * BT, c0))
                with Mesh(("thread",), layout=(BT,), names=("t",)) as tg:
                    t0 = {cs}.rb * BT
                    h = {cs}.hd
                    a_t = tf.schedule((proj[t0:t0 + BT, A_OFF + h:A_OFF + h + 1], tf.empty(Tensor[(BT, 1), "bf16", ((BT @ tg.t, 1), (0, 0)), "rmem"])), op=T.copy())
                    b_t = tf.schedule((proj[t0:t0 + BT, B_OFF + h:B_OFF + h + 1], tf.empty(Tensor[(BT, 1), "bf16", ((BT @ tg.t, 1), (0, 0)), "rmem"])), op=T.copy())
                    al = tf.schedule((a_log[h:h + 1], tf.empty(Tensor[(1,), "bf16", (1,), "rmem"])), op=T.copy())
                    db = tf.schedule((dt_bias[h:h + 1], tf.empty(Tensor[(1,), "bf16", (1,), "rmem"])), op=T.copy())
                    av = tf.cast(a_t, dtype="f32") + tf.cast(db, dtype="f32")
                    sp = tf.where(av > 20.0, av, tf.log(tf.exp(av) + 1.0))
                    gv = tf.neg(tf.exp(tf.cast(al, dtype="f32"))) * sp
                    bv = tf.cast(b_t, dtype="f32")
                    be = tf.cast(tf.cast(1.0 / (tf.exp(tf.neg(bv)) + 1.0), dtype="bf16"), dtype="f32")
                    go = tf.schedule((gv, tf.empty(Tensor[(BT, 1), "f32", (BT, 1)])), op=T.copy())
                    out = tf.insert_slice(out, go, (t0, G_COL + h))
                    bo = tf.schedule((be, tf.empty(Tensor[(BT, 1), "f32", (BT, 1)])), op=T.copy())
                    out = tf.insert_slice(out, bo, (t0, BETA_COL + h))
'''

WINDOW_BODY = '''            with Mesh(cta[NB - 1:NB, :], layout=(1, 80), names=("rb", "hd")) as cw:
                with Mesh(("thread",), layout=(W * 32,), names=("t",)) as tw:
                    wv = tf.schedule((proj[T_ - 3:T_, cw.hd * 128:cw.hd * 128 + 128], tf.empty(Tensor[(3, 128), "bf16", ((3, 128 @ tw.t), (1, 0)), "rmem"])), op=T.copy())
                    wo = tf.schedule((tf.cast(wv, dtype="f32"), tf.empty(Tensor[(3, 128), "f32", (3, 128)])), op=T.copy())
                    out = tf.insert_slice(out, wo, (T_, cw.hd * 128))
'''


def prep_source(T):
    R, W = _prep_config(T)
    taps = "".join(TAP.format(j=j, lhs="acc", prev="" if j == 0 else "acc + ") for j in range(4))
    body = []
    for cs, h0, h1, role in (("cq", 0, 32, QK_BODY), ("cv", 32, 80, V_BODY)):
        body.append(f'            with Mesh(cta[:, {h0}:{h1}], layout=(NB, {h1 - h0}), names=("rb", "hd")) as {cs}:\n')
        body.append(CONV_BODY.format(cs=cs, H0=h0, TAPS=taps.rstrip("\n")))
        body.append(role.format(cs=cs))
    body.append(WINDOW_BODY)
    return PREP.format(T=T, R=R, W=W, BT=R * W, TH=32 * W, PROJ_PAD=PROJ_PAD, PACK=CONV + 96, PRE=PRE, CONV=CONV,
                       A_OFF=A_OFF, B_OFF=B_OFF, EPS=repr(EPS_L2_BF16), BODY="".join(body).rstrip("\n"))


# --------------------------------------------------------------------------------------
# shared layouts (bf16 tiles: 64-column 128B-swizzled panels; f32 tiles: 32-column panels)
# --------------------------------------------------------------------------------------
LAY_ROW128 = "Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)"   # (64, 128) bf16, TMA tile
LAY_ROW64 = "Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)"              # (64, 64) bf16
# views of a LAY_ROW128 tile as a wgmma operand
VIEW_KT_N2 = "Layout(((2, 4, 16), (2, 4, 8)), ((4096, 16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3)"     # (128, 64): B K-major, N in 2 groups
VIEW_T_MN = "Layout(((2, 1, 64), (4, 2, 8)), ((4096, 1024, 1), (1024, 512, 64))) | Swizzle(3, 4, 3)"     # (128, 64): A MN-major (tile^T)
VIEW_MN_N2 = "Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)"    # (64, 128): B MN-major, N in 2 groups
VIEW_64T_N2 = "Layout(((4, 16), (2, 4, 8)), ((16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3)"              # (64, 64)^T: B K-major, N in 2 groups
VIEW_64T = "Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3)"                          # (64, 64)^T: B K-major


def f32_tile(rows, cols):
    """(rows, cols) f32 tile in 32-column 128B-swizzled panels (the TMA box is 32 wide)."""
    return f"Layout((({rows // 8}, 8), ({cols // 32}, 32)), ((256, 32), ({rows * 32}, 1))) | Swizzle(3, 4, 3)"


def frag(c, n):
    """wgmma accumulator of a (64, n) tile on one warpgroup."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, 512, 128, 64))"


def frag_n2(c, n):
    """(64, n) accumulator over two warpgroups, warpgroup g owning columns [g*n/2, (g+1)*n/2)."""
    h = n // 2
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, 2 @ {c}.group, {h // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, {64 * h}, 512, 128, 64))"


def rowv_n2(c):
    """(64, 1): one value per accumulator row of frag_n2, replicated along the columns."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0))"


def colv_n2(c, n):
    """(1, n): one value per accumulator column of frag_n2, replicated along the rows."""
    h = n // 2
    return f"((1, 2 @ {c}.group, {h // 8}, 4 @ {c}.lane4, 2, 1, 1), (0, {64 * h}, 512, 128, 64, 0, 0))"


# --------------------------------------------------------------------------------------
# gdn_chunk
# --------------------------------------------------------------------------------------
CHUNK = '''"""Scheduled HIR: GDN chunk (intra-chunk WY quantities) at prompt length {T} (NC = {NC} chunks).

Implements model.py Qwen3_8DeltaNetLayer.delta_rule, the per-chunk part (everything
before the sequential pass over chunks), for one value head h and chunk c, with
G = in-chunk cumulative sum of g (kept in log2 units, G' = G * log2(e)):
  A   = strict_lower(beta_i * exp(G_i - G_j) * k_i.k_j)
  X   = (I + A)^-1        (the HIR's forward substitution of -A, plus I, is this X)
  U^T = V^T (X diag(beta))^T               (HIR `value`, transposed, f32)
  W   = X diag(beta e^G) K                  (HIR `k_cumdecay`, as bf16 hi + lo)
  P   = lower(q_i.k_j * exp(G_i - G_j)) / sqrt(128)   (HIR `scores` with q scaled, bf16)
  F   = [exp(G_63 - G_j), exp(G_j) / sqrt(128), exp(G_63)]   (the chunk decay factors)
Runtime kernel: runtime/kernels/gdn.py make_chunk({T}) (`gdn_chunk`, precise state path).

Decomposition (as the kernel): CTA mesh (value head h, chunk c), c fastest (tilelang
grid (NC, 48)); key head kh = h // 3. 256 threads = 2 warpgroups.
- One warp issues the TMA loads of the K, Q, V tiles (64 x 128 bf16, two 64-column
  128B-swizzled panels each).
- Warp 0 computes G (lane l owns rows 2l, 2l+1), threads 32..95 load beta.
- KK = K K^T and QK = Q K^T: wgmma SS m64n32k16, K-major, warpgroup g owns columns
  [32g, 32g+32) (the kernel's tilelang partition: N split over the warpgroups).
- In the accumulator registers: dec = exp2(min(G'_i - G'_j, 0)),
  A = beta_i * dec * KK (strictly lower), P = bf16(QK * dec / sqrt(128)) (lower).
  A goes to a padded (64, 68) f32 shared tile, P is staged in shared memory and
  TMA-stored. Threads 0..63 compute exp2(G'), BG = beta * exp2(G') and F (TMA-stored).
- X = (I + A)^-1 in f32 SIMT on shared memory: the four 16x16 diagonal blocks by
  forward substitution (threads 0..63: 16 per block, thread = one column, 16 rows in
  registers), then two doubling levels (X21 = -X22 A21 X11), each as a Y phase into a
  padded Y tile and an X phase:
    level 16 -> 32, blocks (1,0) on warpgroup 0 and (3,2) on warpgroup 1, thread =
      (row, column pair), 1x2 register tile, 16-term sums;
    level 32 -> 64, Y = A[32:64, 0:32] X[0:32, 0:32] with warpgroup g owning columns
      [16g, 16g+16), then X[32:64, 0:32] = -X[32:64, 32:64] Y with warpgroup 0 owning
      rows 48..63 and warpgroup 1 rows 32..47; 2x2 register tiles, 32-term sums (the
      cleared upper blocks of X add exact zeros).
- Mb = X diag(beta), Mg = X diag(beta e^G), each split into bf16 hi + lo, stored
  as 128B-swizzled (64, 64) tiles (256 threads, 4 rows x float4 each).
- U^T = V^T Mbh^T + V^T Mbl^T: wgmma SS m64n32k16 with V^T MN-major from the V tile
  (2 M atoms per warpgroup, warpgroup g owns columns [32g, 32g+32)); W = Mgh K + Mgl K:
  wgmma SS m64n64k16 with K MN-major from the same K tile (warpgroup g owns dk columns
  [64g, 64g+64)). W is split into bf16 hi and lo; U^T, W hi, W lo are staged in
  swizzled shared tiles and TMA-stored.

Differences to the kernel:
- Result: scheduled lowering accepts one tensor result. The kernel's four workspaces
  (U^T (48, NC, 128, 64) f32, W (48, NC, 2, 64, 128) bf16, P (48, NC, 64, 64) bf16,
  F (48, NC, 3, 64) f32) are one f32 tensor (48, NC, {PACK}): per (h, c), [0, 8192) =
  U^T (128 x 64 row-major), [8192, 16384) = W hi (64 x 128), [16384, 24576) = W lo,
  [24576, 28672) = P (64 x 64), [28672, 28864) = F (3 x 64). W hi/lo and P keep their
  bf16 rounding and are widened to f32 exactly before staging, so their staging tiles
  are f32 (32-column panels) where the kernel stages and TMA-stores bf16 tiles.
- G: the kernel scans g with shfl_up (Hillis-Steele over 32 lanes, 2 rows per lane).
  TileFoundry has no shuffle; this program states the inclusive cumsum as the masked
  row sum G_i = sum_j g_j [j <= i] (the HIR's product with the triangular ones), lane l
  computing its rows 2l, 2l+1 from all 64 g values. Same values up to f32 summation
  order.
- Masks: the kernel evaluates i > j, i >= j and i == j from indices
  (T.if_then_else); this program reads them from a constant input `tri` (3, 64, 64)
  f32 ([0] = (j <= i), [1] = (j < i), [2] = identity) and multiplies by them
  (tf.arange has no TIR lowering). Products with 0/1 give the same values (+0 vs -0
  only).
- Broadcast operands (G'_i, G'_j, beta_i in the accumulator layout; exp2(G'_63)) are
  register copies read from the shared G / beta vectors; the kernel reads G_s[i],
  G_s[j], B_s[i] from shared memory inside its element loop. The row and column
  vectors are added to a zero accumulator first so every elementwise operand carries
  the accumulator layout (+ 0.0 changes no value).
- Forward substitution: the kernel subtracts A[i, j] * x[j] term by term from (i == j);
  this program subtracts the row sum; same terms, f32 order differs. The kernel unrolls
  the 16 steps; here they are one loop.
- Inverse level 16 -> 32: the kernel runs the Y phase of both blocks on all 256 threads,
  synchronises, then the X phase; here the Y and X steps of one block are one region on
  its warpgroup (same thread -> element mapping, same values).
- In-place reuse: the kernel's shared allocator aliases buffers whose lifetimes do not
  overlap (e.g. UT_s on K_s, W_s on V_s); this program names every buffer once.
- Synchronisation (mbarrier waits on the TMA loads, __syncthreads between phases,
  wgmma fences/waits) is not stated; TileFoundry derives it from the data flow.
"""
{PRE}T_ = {T}
NC = T_ // 64
HV = 48
D = 128
C = 64
LOG2E = 1.4426950408889634
SCALE = 0.08838834764831845
U_OFF = 0
WH_OFF = 8192
WL_OFF = 16384
P_OFF = 24576
F_OFF = 28672
PACK = {PACK}


@module(entry="gdn_chunk", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", HV * NC), Topology("thread", 256)))
class SCHED_GDN_CHUNK_T{T}:
    @func
    def gdn_chunk(q: Tensor[(T_, 16, D), "bf16"], k: Tensor[(T_, 16, D), "bf16"], v: Tensor[(T_, HV, D), "bf16"],
                  g: Tensor[(T_, HV), "f32"], beta: Tensor[(T_, HV), "bf16"],
                  tri: Tensor[(3, C, C), "f32"]) -> Tensor[(HV, NC, PACK), "f32"]:
        with Mesh(("cta",), layout=(HV, NC), names=("h", "c")) as cta:
            with Mesh(("thread",), layout=(2, 128), names=("warpgroup", "participant")) as threads:
                s_mma = T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                u_mma = T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K)
                w_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(HV, NC, PACK), "f32"])
                kh = cta.h // 3
                t0 = cta.c * C
{BODY}
                return out
'''


def chunk_source(T):
    NC = T // 64
    L = []
    i16, i20, i24 = (" " * n for n in (16, 20, 24))

    def add(ind, s):
        L.append(ind + s)

    def wgs(ind, name="cw"):
        add(ind, f'with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as {name}:')

    # 1. TMA loads; X starts cleared
    add(i16, "with threads[0, :32] as _ld:")
    for nm, src in (("ks", "k[t0:t0 + C, kh:kh + 1, :]"), ("qs", "q[t0:t0 + C, kh:kh + 1, :]"),
                    ("vs", "v[t0:t0 + C, cta.h:cta.h + 1, :]")):
        add(i20, f'{nm} = tf.schedule((tf.reshape({src}, new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", {LAY_ROW128}, "smem"])), op=T.copy_async_tensor())')
    add(i16, 'Xs = tf.zeros(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])')
    add(i16, 'Ys = tf.empty(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])')
    # 2. G (warp 0, lane l: rows 2l, 2l+1) and beta (threads 32..95)
    add(i16, 'with Mesh(threads[0, 0:32], layout=(32,), names=("l",)) as wl:')
    add(i20, 'gr = tf.schedule((tf.reshape(g[t0:t0 + C, cta.h:cta.h + 1], new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", (1, C), "rmem"])), op=T.copy())')
    add(i20, 'lm = tf.schedule((tf.reshape(tri[0:1, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", ((32 @ wl.l, 2, C), (0, C, 1)), "rmem"])), op=T.copy())')
    add(i20, 'G = tf.reduce(lm * gr, axes=(-1,), keepdim=True, kind="sum") * LOG2E')
    add(i20, 'Gs = tf.schedule((G, tf.empty(Tensor[(C, 1), "f32", Layout((C, 1), (1, 1)), "smem"])), op=T.copy())')
    add(i16, 'with Mesh(threads[0, 32:96], layout=(64,), names=("r",)) as wb:')
    add(i20, 'bb = tf.schedule((beta[t0:t0 + C, cta.h:cta.h + 1], tf.empty(Tensor[(C, 1), "bf16", ((64 @ wb.r, 1), (0, 0)), "rmem"])), op=T.copy())')
    add(i20, 'Bs = tf.schedule((tf.cast(bb, dtype="f32"), tf.empty(Tensor[(C, 1), "f32", Layout((C, 1), (1, 1)), "smem"])), op=T.copy())')
    # 3. KK, QK (N split over the two warpgroups)
    wgs(i16)
    for nm, a in (("KK", "ks"), ("QK", "qs")):
        add(i20, f'{nm} = tf.schedule((tf.zeros(Tensor[(C, C), "f32", {frag_n2("cw", 64)}, "rmem"]), {a}, tf.bitcast(ks, {VIEW_KT_N2})), op=T.tiled_mma(atom=s_mma), repeat=(1, 2, 8))')
    # 4. threads 0..63: exp2(G'), BG, F
    add(i16, 'with Mesh(threads[0, 0:64], layout=(64,), names=("r",)) as m64:')
    add(i20, 'gi = tf.schedule((tf.reshape(Gs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"])), op=T.copy())')
    add(i20, 'bi = tf.schedule((tf.reshape(Bs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"])), op=T.copy())')
    add(i20, 'g63 = tf.schedule((Gs[C - 1:C, :], tf.empty(Tensor[(1, 1), "f32", (1, 1), "rmem"])), op=T.copy())')
    add(i20, 'eg = tf.exp2(gi)')
    add(i20, 'BGs = tf.schedule((bi * eg, tf.empty(Tensor[(1, C), "f32", Layout((1, C), (C, 1)), "smem"])), op=T.copy())')
    add(i20, 'Fs = tf.empty(Tensor[(3, C), "f32", Layout((3, C), (C, 1)), "smem"])')
    add(i20, 'Fs = tf.insert_slice(Fs, tf.exp2(tf.neg(gi) + g63), (0, 0))')
    add(i20, 'Fs = tf.insert_slice(Fs, eg * SCALE, (1, 0))')
    add(i20, 'Fs = tf.insert_slice(Fs, tf.zeros(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"]) + tf.exp2(g63), (2, 0))')
    # 5. A and P in the accumulator registers
    wgs(i16)
    zf = f'tf.zeros(Tensor[(C, C), "f32", {frag_n2("cw", 64)}, "rmem"])'
    add(i20, f'Gi = tf.schedule((Gs, tf.empty(Tensor[(C, 1), "f32", {rowv_n2("cw")}, "rmem"])), op=T.copy())')
    add(i20, f'Gj = tf.schedule((tf.reshape(Gs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", {colv_n2("cw", 64)}, "rmem"])), op=T.copy())')
    add(i20, f'Bi = tf.schedule((Bs, tf.empty(Tensor[(C, 1), "f32", {rowv_n2("cw")}, "rmem"])), op=T.copy())')
    add(i20, f'lower_f = tf.schedule((tf.reshape(tri[0:1, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", {frag_n2("cw", 64)}, "rmem"])), op=T.copy())')
    add(i20, f'strict_f = tf.schedule((tf.reshape(tri[1:2, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", {frag_n2("cw", 64)}, "rmem"])), op=T.copy())')
    add(i20, f'dec = tf.exp2(tf.min({zf} + Gi - Gj, 0.0))')
    add(i20, f'Afr = ({zf} + Bi) * dec * KK * strict_f')
    add(i20, 'Pfr = tf.cast(tf.cast(QK * dec * SCALE * lower_f, dtype="bf16"), dtype="f32")')
    add(i20, 'As = tf.schedule((Afr, tf.empty(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])), op=T.copy())')
    add(i20, f'Pst = tf.schedule((Pfr, tf.empty(Tensor[(C, C), "f32", {f32_tile(64, 64)}, "smem"])), op=T.copy())')
    add(i16, "with threads[0, :32] as _st:")
    add(i20, 'po = tf.schedule((Pst, tf.empty(Tensor[(C, C), "f32", (C, C)])), op=T.copy_async_tensor())')
    add(i20, 'out = tf.insert_slice(out, tf.reshape(po, new_shape=(1, 1, C * C)), (cta.h, cta.c, P_OFF))')
    add(i20, 'fo = tf.schedule((Fs, tf.empty(Tensor[(3, C), "f32", (3, C)])), op=T.copy_async_tensor())')
    add(i20, 'out = tf.insert_slice(out, tf.reshape(fo, new_shape=(1, 1, 3 * C)), (cta.h, cta.c, F_OFF))')
    # 6. X = (I + A)^-1: diagonal blocks by forward substitution
    for b in range(4):
        add(i16, f'with Mesh(threads[0, {16 * b}:{16 * b + 16}], layout=(16,), names=("cc",)) as md{b}:')
        add(i20, f'Xb = tf.zeros(Tensor[(16, 16), "f32", ((16, 16 @ md{b}.cc), (1, 0)), "rmem"])')
        add(i20, 'for i in range(16):')
        add(i24, f'ai = tf.schedule((tf.reshape(As[{16 * b} + i:{16 * b} + i + 1, {16 * b}:{16 * b + 16}], new_shape=(16, 1)), tf.empty(Tensor[(16, 1), "f32", (16, 1), "rmem"])), op=T.copy())')
        add(i24, f'ei = tf.schedule((tf.reshape(tri[2:3, i:i + 1, 0:16], new_shape=(1, 16)), tf.empty(Tensor[(1, 16), "f32", ((1, 16 @ md{b}.cc), (0, 0)), "rmem"])), op=T.copy())')
        add(i24, 'Xb = tf.insert_slice(Xb, ei - tf.reduce(Xb * ai, axes=(0,), keepdim=True, kind="sum"), (i, 0))')
        add(i20, f'Xs = tf.insert_slice(Xs, Xb, ({16 * b}, {16 * b}))')
    # level 16 -> 32: block (1,0) on warpgroup 0, block (3,2) on warpgroup 1; thread = (row r, column pair cp)
    for wg, bj in ((0, 0), (1, 2)):
        bi = bj + 1
        m = f"ml{wg}"
        add(i16, f'with Mesh(threads[{wg}, :], layout=(16, 8), names=("r", "cp")) as {m}:')
        add(i20, f'ar = tf.schedule((tf.reshape(As[{16 * bi}:{16 * bi + 16}, {16 * bj}:{16 * bi}], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ {m}.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())')
        add(i20, f'xc = tf.schedule((tf.reshape(Xs[{16 * bj}:{16 * bi}, {16 * bj}:{16 * bj + 16}], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ {m}.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())')
        add(i20, f'Ys = tf.insert_slice(Ys, tf.reduce(ar * xc, axes=(1,), keepdim=False, kind="sum"), ({16 * bi}, {16 * bj}))')
        add(i20, f'dr = tf.schedule((tf.reshape(Xs[{16 * bi}:{16 * bi + 16}, {16 * bi}:{16 * bi + 16}], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ {m}.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())')
        add(i20, f'yc = tf.schedule((tf.reshape(Ys[{16 * bi}:{16 * bi + 16}, {16 * bj}:{16 * bj + 16}], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ {m}.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())')
        add(i20, f'Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), ({16 * bi}, {16 * bj}))')
    # level 32 -> 64, Y phase: Y = A[32:64, 0:32] X[0:32, 0:32]; thread (hw, rp, c8): rows 2rp, 2rp+1, columns 16hw + 2c8 + {0, 1}
    add(i16, 'with Mesh(threads[0:2, :], layout=(2, 16, 8), names=("hw", "rp", "c8")) as m2y:')
    add(i20, 'ar = tf.schedule((tf.reshape(As[32:64, 0:32], new_shape=(32, 32, 1)), tf.empty(Tensor[(32, 32, 1), "f32", ((16 @ m2y.rp, 2, 32, 1), (0, 32, 1, 0)), "rmem"])), op=T.copy())')
    add(i20, 'xc = tf.schedule((tf.reshape(Xs[0:32, 0:32], new_shape=(1, 32, 32)), tf.empty(Tensor[(1, 32, 32), "f32", ((1, 32, 2 @ m2y.hw, 8 @ m2y.c8, 2), (0, 2, 0, 0, 1)), "rmem"])), op=T.copy())')
    add(i20, 'Ys = tf.insert_slice(Ys, tf.reduce(ar * xc, axes=(1,), keepdim=False, kind="sum"), (32, 0))')
    # level 32 -> 64, X phase: X[32:64, 0:32] = -X[32:64, 32:64] Y; warpgroup 0 rows 48..63, warpgroup 1 rows 32..47;
    # thread (r8, cr): rows 2r8, 2r8+1 of its half, columns 2cr, 2cr+1
    for wg, r0 in ((0, 48), (1, 32)):
        m = f"m2x{wg}"
        add(i16, f'with Mesh(threads[{wg}, :], layout=(8, 16), names=("r8", "cr")) as {m}:')
        add(i20, f'dr = tf.schedule((tf.reshape(Xs[{r0}:{r0 + 16}, 32:64], new_shape=(16, 32, 1)), tf.empty(Tensor[(16, 32, 1), "f32", ((8 @ {m}.r8, 2, 32, 1), (0, 32, 1, 0)), "rmem"])), op=T.copy())')
        add(i20, f'yc = tf.schedule((tf.reshape(Ys[32:64, 0:32], new_shape=(1, 32, 32)), tf.empty(Tensor[(1, 32, 32), "f32", ((1, 32, 16 @ {m}.cr, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())')
        add(i20, f'Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), ({r0}, 0))')
    # 7. Mb = X diag(beta), Mg = X diag(beta e^G), bf16 hi + lo
    add(i16, 'with Mesh(("thread",), layout=(16, 16), names=("rr", "cc")) as tp:')
    add(i20, 'xr = tf.schedule((Xs, tf.empty(Tensor[(C, C), "f32", ((4, 16 @ tp.rr, 16 @ tp.cc, 4), (4, 0, 0, 1)), "rmem"])), op=T.copy())')
    add(i20, 'bj = tf.schedule((tf.reshape(Bs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 1, 16 @ tp.cc, 4), (0, 0, 0, 1)), "rmem"])), op=T.copy())')
    add(i20, 'bgj = tf.schedule((BGs, tf.empty(Tensor[(1, C), "f32", ((1, 1, 16 @ tp.cc, 4), (0, 0, 0, 1)), "rmem"])), op=T.copy())')
    add(i20, 'mb = xr * bj')
    add(i20, 'mg = xr * bgj')
    add(i20, 'mbh = tf.cast(mb, dtype="bf16")')
    add(i20, 'mgh = tf.cast(mg, dtype="bf16")')
    for nm, val in (("Mbh", "mbh"), ("Mbl", 'tf.cast(mb - tf.cast(mbh, dtype="f32"), dtype="bf16")'),
                    ("Mgh", "mgh"), ("Mgl", 'tf.cast(mg - tf.cast(mgh, dtype="f32"), dtype="bf16")')):
        add(i20, f'{nm} = tf.schedule(({val}, tf.empty(Tensor[(C, C), "bf16", {LAY_ROW64}, "smem"])), op=T.copy())')
    # 8. U^T and W (wgmma), W hi/lo, staging
    wgs(i16)
    add(i20, f'UT = tf.schedule((tf.zeros(Tensor[(D, C), "f32", ((2, 4 @ cw.warp, 2, 8 @ cw.lane8, 2 @ cw.group, 4, 4 @ cw.lane4, 2), (2048, 16, 8, 1, 4096, 512, 128, 64)), "rmem"]), tf.bitcast(vs, {VIEW_T_MN}), tf.bitcast(Mbh, {VIEW_64T_N2})), op=T.tiled_mma(atom=u_mma), repeat=(2, 2, 4))')
    add(i20, f'UT = tf.schedule((UT, tf.bitcast(vs, {VIEW_T_MN}), tf.bitcast(Mbl, {VIEW_64T_N2})), op=T.tiled_mma(atom=u_mma), repeat=(2, 2, 4))')
    add(i20, f'WF = tf.schedule((tf.zeros(Tensor[(C, D), "f32", {frag_n2("cw", 128)}, "rmem"]), Mgh, tf.bitcast(ks, {VIEW_MN_N2})), op=T.tiled_mma(atom=w_mma), repeat=(1, 2, 4))')
    add(i20, f'WF = tf.schedule((WF, Mgl, tf.bitcast(ks, {VIEW_MN_N2})), op=T.tiled_mma(atom=w_mma), repeat=(1, 2, 4))')
    add(i20, f'UTs = tf.schedule((UT, tf.empty(Tensor[(D, C), "f32", {f32_tile(128, 64)}, "smem"])), op=T.copy())')
    add(i20, 'wh = tf.cast(WF, dtype="bf16")')
    add(i20, 'wlo = tf.cast(WF - tf.cast(wh, dtype="f32"), dtype="bf16")')
    add(i20, f'Whs = tf.schedule((tf.cast(wh, dtype="f32"), tf.empty(Tensor[(C, D), "f32", {f32_tile(64, 128)}, "smem"])), op=T.copy())')
    add(i20, f'Wls = tf.schedule((tf.cast(wlo, dtype="f32"), tf.empty(Tensor[(C, D), "f32", {f32_tile(64, 128)}, "smem"])), op=T.copy())')
    add(i16, "with threads[0, :32] as _st:")
    for nm, src, shp, off in (("uo", "UTs", "(D, C)", "U_OFF"), ("who", "Whs", "(C, D)", "WH_OFF"), ("wlo_", "Wls", "(C, D)", "WL_OFF")):
        add(i20, f'{nm} = tf.schedule(({src}, tf.empty(Tensor[{shp}, "f32", {shp}])), op=T.copy_async_tensor())')
        add(i20, f'out = tf.insert_slice(out, tf.reshape({nm}, new_shape=(1, 1, C * D)), (cta.h, cta.c, {off}))')
    return CHUNK.format(T=T, NC=NC, PACK=28864, PRE=PRE, BODY="\n".join(L))


# --------------------------------------------------------------------------------------
# gdn_scan
# --------------------------------------------------------------------------------------
SCAN = """\"\"\"Scheduled HIR: GDN scan, plain variant (sequential pass over chunks) at prompt length {T} (NC = {NC}).

Implements model.py Qwen3_8DeltaNetLayer.delta_rule, the pass over the chunks: the
recurrent state across chunks, the read-out o (landed in bf16) and the final state.
The gated RMSNorm of gated_delta_net is a separate kernel (scheduled/gdn/gdn_norm_T{T}.py).
Runtime kernel: runtime/kernels/gdn.py make_scan({T}, 16, {PROJ_PAD}, gated=False)
(`gdn_scan_plain`, the z=None path of GdnKernels.core; precise state path).

Decomposition (as the kernel): CTA mesh (value head h, dv half vb), vb fastest (tilelang
grid (2, 48) of independent CTAs, no cluster). Each CTA of the pair owns 64 of the
128 dv columns independently: CTA vb holds S^T rows [64vb, 64vb+64) (64 dv x 128 dk,
f32) in the wgmma accumulator of warpgroup 0 and writes only its own read-out columns
and state columns; no data crosses between the two CTAs. 384 threads = 3 warpgroups:
- warpgroup 0 (math), per chunk c, with bf16 hi + lo splits S = Sh + Sl, W = Wh + Wl:
    S^T *= exp(G_63)                      (F[2, 0], scaled before any wgmma)
    Y = Sh Wh^T + Sh Wl^T + Sl Wh^T       (wgmma RS m64n64k16, S from registers)
    O = Sh Q^T                            (wgmma RS m64n64k16)
    Delta^T = U^T - Y (f32), Dk = Delta^T * exp(G_63 - G_j) split bf16 hi + lo
    S^T += Dh K + Dl K                    (wgmma RS, K MN-major from the K tile)
    O = O * exp(G_j) / sqrt(128) + bf16(Delta^T) P^T   (wgmma RS m64n64k16)
    bf16(O) -> shared tile O_s[t, dv] (transposed, rows padded to 72, 2 buffers).
  The bf16 S, Delta and Dk operands are restated from the accumulator layout into the
  wgmma A-operand register layout (k-tile mode first) by register copies.
- warpgroup 1, one warp: TMA loads per chunk into 2-deep rings: W hi, W lo, Q, K
  (64 x 128 bf16, 128B-swizzled panels), P (64 x 64 bf16), U^T rows of this dv half
  (64 x 64 f32, swizzled), F (3 x 64 f32).
- warpgroups 1 + 2 (store): thread (half, r, p) copies row t = 32*half + r of O_s,
  dv columns 16p .. 16p + 15, to the read-out (64 values per row and CTA).
After the last chunk warpgroup 0 stores S^T transposed (state[h, dk, 64vb + dv]).

Differences to the kernel:
- State update instruction: the kernel issues one m64n128k16 per k16 step whose B
  descriptor spans both 64-column panels of the K tile (panel stride 4096 elements,
  LBO 8192 B). TileFoundry's MN-major B pattern requires the panel stride to be
  16 * 64 elements and refuses the kernel's tile ("rhs: could not satisfy lb == 16 *
  Wb"), so S^T += D K is stated as two m64n64k16 atoms per k16 step (one per panel,
  repeat (1, 2, 4)) on the same K tile; the accumulator layout lists the N tile mode
  first. Same products, same f32 accumulation per element.
- Pipelining: the kernel stores the read-out one chunk behind the math (store of chunk
  c-1 in iteration c) and runs the math warpgroup ahead through mbarrier-guarded
  double buffers (full / empty, ofull / oempty). This program states one chunk per
  loop iteration with 2-deep rings (buffers=2); the overlap is left to lowering.
- Register reallocation (setmaxnreg 240 / 128) and warp-specialisation barriers are
  not stated (no TileFoundry construct).
- Result: scheduled lowering accepts one tensor result. The read-out (T, 48, 128)
  bf16 and the final state (48, 128, 128) f32 are one f32 tensor (T + 128, 48, 128):
  rows 0..T-1 = read-out (bf16 values widened exactly), row T + dk = state[:, dk, :].
  The kernel writes the read-out as bf16 and the state as f32 into two buffers.
- Inputs: the kernel's workspaces as it reads them (U^T f32 (48, NC, 128, 64), W bf16
  (48, NC, 2, 64, 128), P bf16 (48, NC, 64, 64), F f32 (48, NC, 3, 64)); the
  scheduled gdn_chunk states them packed in one f32 tensor (see its header).
\"\"\"
{PRE}T_ = {T}
NC = T_ // 64
HV = 48
D = 128
C = 64
S_ROW = T_


@module(entry="gdn_scan", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", HV * 2), Topology("thread", 384)))
class SCHED_GDN_SCAN_T{T}:
    @func
    def gdn_scan(q: Tensor[(T_, 16, D), "bf16"], k: Tensor[(T_, 16, D), "bf16"], ut: Tensor[(HV, NC, D, C), "f32"],
                 wb: Tensor[(HV, NC, 2, C, D), "bf16"], pb: Tensor[(HV, NC, C, C), "bf16"],
                 fc: Tensor[(HV, NC, 3, C), "f32"]) -> Tensor[(T_ + 128, HV, D), "f32"]:
        with Mesh(("cta",), layout=(HV, 2), names=("h", "vb")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                rs_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                st_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(T_ + 128, HV, D), "f32"])
                kh = cta.h // 3
                d0 = cta.vb * 64
{BODY}
                return out
"""


def afrag(c, ktiles):
    """bf16 wgmma A operand in registers for a (64, 16*ktiles) tile: the k-tile mode first."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, {ktiles}, 2, 4 @ {c}.lane4, 2), (16, 8, 1, 1024, 512, 128, 64))"


def colv1(c, n):
    """(1, n): one value per accumulator column of frag(c, n), replicated along the rows."""
    return f"((1, {n // 8}, 4 @ {c}.lane4, 2, 1, 1), (0, 512, 128, 64, 0, 0))"


SCAN_VIEW_KT = "Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3)"     # (128, 64): B K-major view of a (64, 128) tile
SCAN_VIEW_MN = VIEW_MN_N2   # (64, 128): B MN-major, two n = 64 atoms (one per 64-column panel)


def frag_2n(c, n):
    """(64, 2n) accumulator on one warpgroup issued as two n-wide atoms (the N tile mode first)."""
    return f"((4 @ {c}.warp, 2, 8 @ {c}.lane8, 2, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, {64 * n}, 512, 128, 64))"


def scan_source(T):
    NC = T // 64
    L = []
    i16, i20, i24 = (" " * n for n in (16, 20, 24))

    def add(ind, s):
        L.append(ind + s)

    math = 'with Mesh(threads[0:1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as cm:'
    add(i16, math)
    add(i20, f'St = tf.zeros(Tensor[(C, D), "f32", {frag_2n("cm", 64)}, "rmem"])')
    add(i16, "for c in range(NC):")
    # producer: one warp of warpgroup 1
    add(i20, "with threads[1, :32] as _ld:")
    loads = (
        ("W_s", "tf.reshape(wb[cta.h:cta.h + 1, c:c + 1, 0:1, :, :], new_shape=(C, D))", "(C, D)", '"bf16"', LAY_ROW128),
        ("Wl_s", "tf.reshape(wb[cta.h:cta.h + 1, c:c + 1, 1:2, :, :], new_shape=(C, D))", "(C, D)", '"bf16"', LAY_ROW128),
        ("Q_s", "tf.reshape(q[c * C:c * C + C, kh:kh + 1, :], new_shape=(C, D))", "(C, D)", '"bf16"', LAY_ROW128),
        ("K_s", "tf.reshape(k[c * C:c * C + C, kh:kh + 1, :], new_shape=(C, D))", "(C, D)", '"bf16"', LAY_ROW128),
        ("P_s", "tf.reshape(pb[cta.h:cta.h + 1, c:c + 1, :, :], new_shape=(C, C))", "(C, C)", '"bf16"', LAY_ROW64),
        ("U_s", "tf.reshape(ut[cta.h:cta.h + 1, c:c + 1, d0:d0 + 64, :], new_shape=(C, C))", "(C, C)", '"f32"', f32_tile(64, 64)),
        ("F_s", "tf.reshape(fc[cta.h:cta.h + 1, c:c + 1, :, :], new_shape=(3, C))", "(3, C)", '"f32"', "Layout((3, C), (C, 1))"),
    )
    for nm, src, shp, dt, lay in loads:
        add(i24, f'{nm} = tf.schedule(({src}, tf.empty(Tensor[{shp}, {dt}, {lay}, "smem"], buffers=2)), op=T.copy_async_tensor())')
    # math warpgroup
    add(i20, math)
    f64 = frag("cm", 64)
    add(i24, 'sbv = tf.cast(St, dtype="bf16")')
    add(i24, f'Sh = tf.schedule((sbv, tf.empty(Tensor[(C, D), "bf16", {afrag("cm", 8)}, "rmem"])), op=T.copy())')
    add(i24, f'Sl = tf.schedule((tf.cast(St - tf.cast(sbv, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, D), "bf16", {afrag("cm", 8)}, "rmem"])), op=T.copy())')
    add(i24, 'eg = tf.schedule((F_s[2:3, 0:1], tf.empty(Tensor[(1, 1), "f32", (1, 1), "rmem"])), op=T.copy())')
    add(i24, 'St = St * eg')
    add(i24, f'Y = tf.schedule((tf.zeros(Tensor[(C, C), "f32", {f64}, "rmem"]), Sh, tf.bitcast(W_s, {SCAN_VIEW_KT})), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))')
    add(i24, f'Y = tf.schedule((Y, Sh, tf.bitcast(Wl_s, {SCAN_VIEW_KT})), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))')
    add(i24, f'Y = tf.schedule((Y, Sl, tf.bitcast(W_s, {SCAN_VIEW_KT})), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))')
    add(i24, f'O = tf.schedule((tf.zeros(Tensor[(C, C), "f32", {f64}, "rmem"]), Sh, tf.bitcast(Q_s, {SCAN_VIEW_KT})), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))')
    add(i24, f'u = tf.schedule((U_s, tf.empty(Tensor[(C, C), "f32", {f64}, "rmem"])), op=T.copy())')
    add(i24, f'f0 = tf.schedule((F_s[0:1, :], tf.empty(Tensor[(1, C), "f32", {colv1("cm", 64)}, "rmem"])), op=T.copy())')
    add(i24, f'f1 = tf.schedule((F_s[1:2, :], tf.empty(Tensor[(1, C), "f32", {colv1("cm", 64)}, "rmem"])), op=T.copy())')
    add(i24, 'dlt = u - Y')
    add(i24, 'dk = dlt * f0')
    add(i24, 'dkh = tf.cast(dk, dtype="bf16")')
    add(i24, f'Dt = tf.schedule((tf.cast(dlt, dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", {afrag("cm", 4)}, "rmem"])), op=T.copy())')
    add(i24, f'Dk = tf.schedule((dkh, tf.empty(Tensor[(C, C), "bf16", {afrag("cm", 4)}, "rmem"])), op=T.copy())')
    add(i24, f'Dkl = tf.schedule((tf.cast(dk - tf.cast(dkh, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", {afrag("cm", 4)}, "rmem"])), op=T.copy())')
    add(i24, f'St = tf.schedule((St, Dk, tf.bitcast(K_s, {SCAN_VIEW_MN})), op=T.tiled_mma(atom=st_mma), repeat=(1, 2, 4))')
    add(i24, f'St = tf.schedule((St, Dkl, tf.bitcast(K_s, {SCAN_VIEW_MN})), op=T.tiled_mma(atom=st_mma), repeat=(1, 2, 4))')
    add(i24, f'O = tf.schedule((O * f1, Dt, tf.bitcast(P_s, {VIEW_64T})), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 4))')
    add(i24, 'O_s = tf.schedule((tf.cast(O, dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", Layout((C, C), (1, 72)), "smem"], buffers=2)), op=T.copy())')
    # read-out store: warpgroups 1 + 2, thread (half, r, p) = row 32*half + r, 16 dv columns
    add(i20, 'with Mesh(threads[1:3, :], layout=(2, 32, 4), names=("half", "r", "p")) as ce:')
    row = "((2 @ ce.half, 32 @ ce.r, 4 @ ce.p, 16), (0, 0, 0, 1))"
    add(i24, f'ov = tf.schedule((tf.bitcast(O_s, Layout((C, C), (72, 1))), tf.empty(Tensor[(C, C), "bf16", {row}, "rmem"])), op=T.copy())')
    add(i24, 'yo = tf.schedule((tf.cast(ov, dtype="f32"), tf.empty(Tensor[(C, C), "f32", (C, C)])), op=T.copy())')
    add(i24, 'out = tf.insert_slice(out, tf.reshape(yo, new_shape=(C, 1, C)), (c * C, cta.h, d0))')
    # final state, transposed
    add(i16, math)
    add(i20, 'sto = tf.schedule((St, tf.empty(Tensor[(C, D), "f32", Layout((C, D), (1, C))])), op=T.copy())')
    add(i20, 'out = tf.insert_slice(out, tf.reshape(tf.bitcast(sto, Layout((D, C), (C, 1))), new_shape=(D, 1, C)), (S_ROW, cta.h, d0))')
    return SCAN.format(T=T, NC=NC, PRE=PRE, PROJ_PAD=PROJ_PAD, BODY="\n".join(L))


# --------------------------------------------------------------------------------------
# gdn_norm
# --------------------------------------------------------------------------------------
NORM = """\"\"\"Scheduled HIR: GDN gated RMSNorm at prompt length {T}.

Implements model.py Qwen3_8DeltaNetLayer.gated_delta_net, the Qwen3_5RMSNormGated part:
  rf = f32(read); ms = mean(rf^2) over the 128 values of a head;
  normed = bf16(bf16(rf * rsqrt(ms + 1e-6)) * gamma); y = bf16(f32(normed) * silu(f32(z))).
Runtime kernel: runtime/kernels/gdn.py make_gated_norm({T}, {PROJ_PAD}) (`gdn_gated_norm`, RW = 1,
WARPS = 4); it reads the plain scan's bf16 read-out (T, 48, 128) and z = in-projection
columns [10240, 16384) through a strided view of the (T, {PROJ_PAD}) buffer.

Decomposition (as the kernel): one CTA per 8 consecutive (t, head) rows of 128 values:
CTA b = 6t + hb (hb fastest; tilelang grid (6T,)) owns token t, heads 8hb .. 8hb + 7.
128 threads = 4 warps x 2 half-warps x 16 lanes; half-warp s of warp w serves row
2w + s; lane l owns channels 8l .. 8l + 7 (16-byte loads of o, z; its 8 gamma values in
registers). Per row: sum of rf^2 over the row (in-thread + across the 16 lanes),
* (1/128), rsqrt(ms + 1e-6), bf16(rf * r), bf16(* gamma), * zf / (exp(-zf) + 1) in f32,
landed in bf16, row store.

Differences to the kernel:
- Row sum: the kernel reproduces torch's summation order for rf.square().mean(-1)
  (chunks of 4 as ((x0^2 + x1^2) + x2^2) + x3^2, shfl_xor 8, 4, 2, 1 per chunk, then the
  lane's two chunks), with IEEE adds/multiplies (no FMA contraction), so it is bitwise
  equal to the HIR. TileFoundry has no shuffle and no way to state the summation order:
  the sum is one tf.reduce over the lane-split axis (same terms, the order is the
  lowering's; equal up to f32 rounding order).
- No FMA contraction: the kernel uses T.ieee_add / T.ieee_mul / T.ieee_div; this
  program states plain binary ops and does not state a contraction policy.
- Gate: the kernel computes n2 * silu(z) with ex2.approx / rcp.approx for the 8 values of
  a lane and one branch per thread redoes n2 * (z / (1 + expf(-z))) in IEEE f32 for every
  value whose fast result is within 16 f32 ulps of a bf16 rounding midpoint or whose
  z < -16 (checked over all finite bf16 (n2, z) pairs); its output is bitwise that of the
  exact formula stated here. The kernel's bf16 roundings are packed vector casts.
- ms = sum * 0.0078125 is the kernel's * (1/128) (a literal `(1.0 / 128.0)` inside an
  expression is refused: "dimension uses unsupported binary operation BinaryKind.DIV").
- Result: the bf16 y (T, 48, 128) itself (one tensor result, no packing).
\"\"\"
{PRE}T_ = {T}
HV = 48
D = 128
PROJ = {PROJ_PAD}
Z_OFF = 10240
RPC = 8
INV_D = 0.0078125
EPS = 1e-6


@module(entry="gdn_norm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", T_ * 6), Topology("thread", 128)))
class SCHED_GDN_NORM_T{T}:
    @func
    def gdn_norm(o: Tensor[(T_, HV, D), "bf16"], proj: Tensor[(T_, PROJ), "bf16"],
                 gamma: Tensor[(D,), "bf16"]) -> Tensor[(T_, HV, D), "bf16"]:
        with Mesh(("cta",), layout=(T_, 6), names=("t", "hb")) as cta:
            h0 = cta.hb * RPC
            with Mesh(("thread",), layout=(4, 2, 16), names=("warp", "half", "ln")) as th:
                out = tf.zeros(Tensor[(T_, HV, D), "bf16"])
                ot = tf.schedule((tf.reshape(o[cta.t:cta.t + 1, h0:h0 + RPC, :], new_shape=(RPC, D)), tf.empty(Tensor[(RPC, D), "bf16", ((4 @ th.warp, 2 @ th.half, 16 @ th.ln, 8), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                zt = tf.schedule((tf.reshape(proj[cta.t:cta.t + 1, Z_OFF + h0 * D:Z_OFF + h0 * D + RPC * D], new_shape=(RPC, D)), tf.empty(Tensor[(RPC, D), "bf16", ((4 @ th.warp, 2 @ th.half, 16 @ th.ln, 8), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                gm = tf.cast(tf.schedule((tf.reshape(gamma, new_shape=(1, D)), tf.empty(Tensor[(1, D), "bf16", ((1, 16 @ th.ln, 8), (0, 0, 1)), "rmem"])), op=T.copy()), dtype="f32")
                rf = tf.cast(ot, dtype="f32")
                rr = tf.rsqrt(tf.reduce(rf * rf, axes=(-1,), keepdim=True, kind="sum") * INV_D + EPS)
                nb = tf.cast(tf.cast(rf * rr, dtype="bf16"), dtype="f32")
                n2 = tf.cast(tf.cast(nb * gm, dtype="bf16"), dtype="f32")
                zf = tf.cast(zt, dtype="f32")
                y = tf.cast(n2 * (zf / (tf.exp(tf.neg(zf)) + 1.0)), dtype="bf16")
                yo = tf.schedule((y, tf.empty(Tensor[(RPC, D), "bf16", (RPC, D)])), op=T.copy())
                out = tf.insert_slice(out, tf.reshape(yo, new_shape=(1, RPC, D)), (cta.t, h0, 0))
            return out
"""


def norm_source(T):
    return NORM.format(T=T, PRE=PRE, PROJ_PAD=PROJ_PAD)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
KERNELS = {"prep": prep_source, "chunk": chunk_source, "scan": scan_source, "norm": norm_source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lengths", type=int, nargs="*", default=[128, 256, 512, 1024])
    ap.add_argument("--only", choices=sorted(KERNELS), default=None)
    ap.add_argument("--out", default=None, help="write scheduled/ and tir/ files under this directory instead")
    args = ap.parse_args()
    base = Path(args.out) if args.out else ROOT
    sdir, tdir = base / "scheduled/gdn", base / "tir/gdn"
    sdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    fails = 0
    for T in args.lengths:
        for kind, fn in KERNELS.items():
            if args.only and kind != args.only:
                continue
            name = f"gdn_{kind}_T{T}"
            (sdir / f"{name}.py").write_text(fn(T))
            argv = ["schedule", "finalize", str(sdir / f"{name}.py"), str(tdir / f"{name}.py")]
            r = subprocess.run([TILEFOUNDRY, *argv], capture_output=True, text=True)
            deep = "maximum recursion depth exceeded" in r.stderr
            if deep:
                # Programs with many mesh regions walk deeper than CPython's default recursion
                # limit; tools/tfdeep.py runs the same CLI entry point on a deep stack.
                r = subprocess.run([sys.executable, str(ROOT / "tools/tfdeep.py"), *argv],
                                   capture_output=True, text=True)
            ok = r.returncode == 0
            fails += not ok
            msg = (r.stderr + r.stdout).strip()
            # rc 0 with an ISL diagnostic on stderr ("isl_map.c:8903: incompatible spaces") also
            # happens for the existing scheduled/gemm/gate_up_T1024.py; it is printed, not fatal.
            note = "" if not msg else ("\n    " if not ok else "  [stderr: ") + msg[:900].replace("\n", "\n    ") + ("" if not ok else "]")
            print(f"{name:18s} finalize {'OK' if ok else 'FAIL'}" + (" (tools/tfdeep.py: default stack hit the recursion limit)" if deep else "") + note, flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
