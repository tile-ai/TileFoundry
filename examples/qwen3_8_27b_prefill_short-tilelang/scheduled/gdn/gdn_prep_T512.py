"""Scheduled HIR: GDN prep at prompt length 512 (R = 4 rows per warp, W = 4 warps).

Implements model.py Qwen3_8DeltaNetLayer.gated_delta_net up to the delta rule:
  causal_conv (depthwise, f32 accumulation in tap order, landed in bf16, silu),
  l2_normalise of the q and k heads (bf16 arithmetic), the gates
  g = -exp(a_log) * softplus(a + dt_bias) (f32) and beta = sigmoid(b) (bf16),
  and the conv window (the last 3 rows of `mixed`).
Runtime kernel: runtime/kernels/gdn.py make_prep(512, 16640) (`gdn_prep`), R, W = _prep_config(512).

Decomposition (as the kernel): CTA mesh (row block rb, head column hd), hd fastest
(tilelang grid (80, NB)); a CTA owns BT = R * W = 16 rows of one 128-channel column
of the conv channels (hd < 16: a q head, hd < 32: a k head, else a v head).
128 threads = W warps x 32 lanes; warp w owns rows w*R .. w*R + R - 1 of the block,
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
  tensor (T + 3, 10336): rows 0..T-1, columns 0..10239 = q | k | v in conv-channel
  order (column 128*hd + c is channel c of head column hd), column 10240 + h = g[:, h],
  column 10288 + h = beta[:, h]; rows T..T+2, columns 0..10239 = window^T (row T + j
  = mixed row T-3+j). The kernel writes q, k, v, beta, window as bf16 into six
  buffers (window transposed (10240, 3)) and g as f32; every bf16 value here is that
  bf16 value widened exactly (the f32 slot is storage only, not a rounding change).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 512
R = 4
W = 4
BT = R * W
NB = T_ // BT
CONV = 10240
PROJ = 16640
A_OFF = 16432
B_OFF = 16384
G_COL = CONV
BETA_COL = CONV + 48
PACK = CONV + 96
EPS_L2 = 9.98377799987793e-07


@module(entry="gdn_prep", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NB * 80), Topology("thread", W * 32)))
class SCHED_GDN_PREP_T512:
    @func
    def gdn_prep(proj: Tensor[(T_, PROJ), "bf16"], conv_w: Tensor[(CONV, 4), "bf16", Layout((CONV, 4), (4, 1))],
                 a_log: Tensor[(48,), "bf16"], dt_bias: Tensor[(48,), "bf16"],
                 rowidx: Tensor[(NB * (BT + 3),), "i64"]) -> Tensor[(T_ + 3, PACK), "f32"]:
        with Mesh(("cta",), layout=(NB, 80), names=("rb", "hd")) as cta:
            out = tf.zeros(Tensor[(T_ + 3, PACK), "f32"])
            with Mesh(cta[:, 0:32], layout=(NB, 32), names=("rb", "hd")) as cq:
                c0 = (cq.hd + 0) * 128
                with Mesh(("thread",), layout=(W, 32), names=("warp", "lane")) as th:
                    rows = rowidx[cq.rb * (BT + 3):cq.rb * (BT + 3) + BT + 3]
                    xs = tf.schedule((proj[:, c0:c0 + 128], tf.empty(Tensor[(BT + 3, 128), "bf16", (BT + 3, 128), "smem"]), rows), op=T.copy_async(fill=0.0))
                    taps = tf.bitcast(conv_w, Layout((4, CONV), (1, 4)))
                    wr = tf.cast(tf.schedule((taps[:, c0:c0 + 128], tf.empty(Tensor[(4, 128), "bf16", ((4, 32 @ th.lane, 4), (1, 0, 4)), "rmem"])), op=T.copy()), dtype="f32")
                    x0 = tf.schedule((xs[0:0 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = tf.cast(x0, dtype="f32") * wr[0:0 + 1, :]
                    x1 = tf.schedule((xs[1:1 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x1, dtype="f32") * wr[1:1 + 1, :]
                    x2 = tf.schedule((xs[2:2 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x2, dtype="f32") * wr[2:2 + 1, :]
                    x3 = tf.schedule((xs[3:3 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x3, dtype="f32") * wr[3:3 + 1, :]
                    xc = tf.cast(tf.cast(acc, dtype="bf16"), dtype="f32")
                    sv = tf.cast(xc / (tf.exp(tf.neg(xc)) + 1.0), dtype="bf16")
                    sf = tf.cast(sv, dtype="f32")
                    sq = tf.cast(tf.cast(sf * sf, dtype="bf16"), dtype="f32")
                    ssum = tf.cast(tf.reduce(sq, axes=(-1,), keepdim=True, kind="sum"), dtype="bf16")
                    se = tf.cast(tf.cast(ssum, dtype="f32") + EPS_L2, dtype="bf16")
                    rr = tf.cast(tf.cast(tf.rsqrt(tf.cast(se, dtype="f32")), dtype="bf16"), dtype="f32")
                    y = tf.cast(tf.cast(sf * rr, dtype="bf16"), dtype="f32")
                    yo = tf.schedule((y, tf.empty(Tensor[(BT, 128), "f32", (BT, 128)])), op=T.copy())
                    out = tf.insert_slice(out, yo, (cq.rb * BT, c0))
            with Mesh(cta[:, 32:80], layout=(NB, 48), names=("rb", "hd")) as cv:
                c0 = (cv.hd + 32) * 128
                with Mesh(("thread",), layout=(W, 32), names=("warp", "lane")) as th:
                    rows = rowidx[cv.rb * (BT + 3):cv.rb * (BT + 3) + BT + 3]
                    xs = tf.schedule((proj[:, c0:c0 + 128], tf.empty(Tensor[(BT + 3, 128), "bf16", (BT + 3, 128), "smem"]), rows), op=T.copy_async(fill=0.0))
                    taps = tf.bitcast(conv_w, Layout((4, CONV), (1, 4)))
                    wr = tf.cast(tf.schedule((taps[:, c0:c0 + 128], tf.empty(Tensor[(4, 128), "bf16", ((4, 32 @ th.lane, 4), (1, 0, 4)), "rmem"])), op=T.copy()), dtype="f32")
                    x0 = tf.schedule((xs[0:0 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = tf.cast(x0, dtype="f32") * wr[0:0 + 1, :]
                    x1 = tf.schedule((xs[1:1 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x1, dtype="f32") * wr[1:1 + 1, :]
                    x2 = tf.schedule((xs[2:2 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x2, dtype="f32") * wr[2:2 + 1, :]
                    x3 = tf.schedule((xs[3:3 + BT, :], tf.empty(Tensor[(BT, 128), "bf16", ((W @ th.warp, R, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    acc = acc + tf.cast(x3, dtype="f32") * wr[3:3 + 1, :]
                    xc = tf.cast(tf.cast(acc, dtype="bf16"), dtype="f32")
                    sv = tf.cast(xc / (tf.exp(tf.neg(xc)) + 1.0), dtype="bf16")
                    vo = tf.schedule((tf.cast(sv, dtype="f32"), tf.empty(Tensor[(BT, 128), "f32", (BT, 128)])), op=T.copy())
                    out = tf.insert_slice(out, vo, (cv.rb * BT, c0))
                with Mesh(("thread",), layout=(BT,), names=("t",)) as tg:
                    t0 = cv.rb * BT
                    h = cv.hd
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
            with Mesh(cta[NB - 1:NB, :], layout=(1, 80), names=("rb", "hd")) as cw:
                with Mesh(("thread",), layout=(W * 32,), names=("t",)) as tw:
                    wv = tf.schedule((proj[T_ - 3:T_, cw.hd * 128:cw.hd * 128 + 128], tf.empty(Tensor[(3, 128), "bf16", ((3, 128 @ tw.t), (1, 0)), "rmem"])), op=T.copy())
                    wo = tf.schedule((tf.cast(wv, dtype="f32"), tf.empty(Tensor[(3, 128), "f32", (3, 128)])), op=T.copy())
                    out = tf.insert_slice(out, wo, (T_, cw.hd * 128))
            return out
