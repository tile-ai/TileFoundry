"""`Qwen3_5LinearAttention`: the Gated-DeltaNet mixer at one token per step.

Three kernels, not eight
------------------------
A decode step of this mixer moves 71 MB of weights and state (33.5 MB qkv +
16.8 MB z + 16.8 MB out + 2 MB state read + 2 MB state write) and about 6 MFLOP
of arithmetic. So the only two questions are *how many bytes* and *how many SMs
are pulling them*.

**Measure the bytes out of HBM, not out of L2.** This machine's L2 is 50 MB, so
`gemv(2048, 8192)`'s 33.5 MB of weights is L2-resident once a benchmark has run
it twice: repeating one call reports 7.4 us / 4.5 TB/s, and rotating eight
separate weight copies through the same graph reports **10.4 us / 3.2 TB/s**.
Every number below is the rotated one, and 3.2 TB/s is the best rate seen on
this machine for any tiling of any of these shapes -- so the floor for 71 MB is
~22 us, not the ~16 us that 4.4 TB/s would suggest.

A `T.gemm` GEMV block sustains **~49 GB/s of HBM** (K=2048, BN=128, 4 stages:
512 KB of weight tile in 10.4 us), so line rate needs ~65 resident blocks. A
GEMV's block count is `N / BN`, which is why `w_out` (K=4096, N=2048, 16 blocks)
measures 1.26 TB/s on its own and why `basic.gemv` cannot be used for any
projection here: two of the three have an `N` too small to fill the machine.
Every projection below is tiled to ~128 resident blocks, splitting K where `N`
alone will not do it.

    1  `_in_proj`     rms_norm(hidden) + qkv + (b, a)     64 + 8 blocks   12.2 us
    2  `_conv_delta`  z + conv + l2norm + delta rule       128 blocks     10.3 us
    3  `_out_proj`    gated output norm + out projection    128 blocks      6.8 us

The seams are forced, not chosen:

* **The conv cannot join kernel 1.** `mixed` needs the *whole* `entry` column
  that kernel 1's 64 blocks jointly produce; there is no grid-wide barrier.
* **`z` cannot join kernel 1.** Two `T.gemm` calls in one kernel do not
  compile (see "What tilelang refused" below), and `z` in kernel 2 costs nothing
  extra anyway: `z` on its own is 8.5 us at 32 blocks and 6.9 us at 128, and
  kernel 2 already has to have 128 blocks for the delta rule.
* **The gated output norm cannot join kernel 2.** It is an RMS over all 128
  value dims of one head, and kernel 2 splits those 128 dims across 4 blocks
  (see `_conv_delta`). Kernel 3 has to read `read` anyway, so it normalises
  there: its K-split is 512 = exactly 4 whole heads, so a block holds every
  value dim of every head it touches.

Why the delta rule parallelises over *value* columns
----------------------------------------------------
`read[j] = sum_i updated[i,j] * q[i]` reduces over the **key** axis, and
`updated` must be written whole. Splitting the key axis would need a cross-block
reduction to get `recalled` before `delta` -- and `delta` gates the write. But a
value column `j` is self-contained: `recalled[j]`, `delta[j]`, `updated[:,j]`
and `read[j]` all need only `S[:,j]`, the full key axis of that one column. So
blocks are `(value head, 32-column group)` = 32*4 = 128, no communication, and
`S[h,:,j0:j0+32]` is 128 B per row -- one whole sector group. One block per head
(32 blocks) measured 5.05 us against 2.74 us for 128; 256 blocks (16 columns)
went back up to 2.98 us on the narrower rows. The 4 MB of state fits in L2, so
this one really is the L2 number, and `delta_step` is close to launch-bound.

Accuracy: the free hi/lo split
------------------------------
`basic.py` leaves MMA rows 1..15 uninitialised because nothing reads them. Row 1
is worth using: putting `bf16(v)` in row 0 and `bf16(v - f32(bf16(v)))` in row 1
makes the epilogue `out[0,j] + out[1,j]` a double-bf16 dot product, which drops
the activation's rounding error from **2.15e-3 to 3.4e-6** for +0.3 us on
K=2048,N=8192. The MMA already did 16x the necessary flops; this spends one more
of the 16 rows and no extra HBM traffic at all.

f32 weights are accepted by *compiling a second variant*, not by casting on the
host (a host cast would be a 50 MB copy on every one of the 30 calls per token).
The variant reads the f32 tile and stages **two** bf16 tiles, high and low, and
issues two `T.gemm`s against the same accumulator. End to end that variant
agrees with f32 torch to 7.5e-6 where a plain round-to-bf16 would give 1.7e-3.
It costs 2x the HBM traffic, which is exactly why the deployment path is bf16;
the f32 path exists so the comparison against the authored `model.py` module is
measuring *this file* and not bf16.

bf16 weights against an f32 reference stay at 4.1e-3 no matter what this file
does: 8 mantissa bits give a K=2048 dot product ~1.7e-3 of relative error, and
`out` is three such projections chained through a softplus. Against a torch
reference holding *the same rounded weights* -- which is the only comparison
that says anything about the kernel -- it is 4.0e-6. The self-test prints both.

What tilelang refused
---------------------
* **Two `T.gemm` calls in two different `T.Pipelined` loops in one kernel.**
  This is the direct route to fusing qkv and z (branch on `blockIdx`, one grid
  of 96 tiles over a 12288-wide concatenated output). Sharing the weight tile
  buffer between the two loops gives
  `InternalError: Check failed: (az->CanProveEqual(input_shape_product *
  rescale_num, shape_product * rescale_den)) is false: InputShape() =
  (4, 128, 128) shape = (4, 4, 128, 128), rescale_num = 16, rescale_den = 16`
  -- the buffer is multi-buffered once per loop, so the second pass tries to
  stage the already-staged `(4, 128, 128)`. Giving each loop its own tile buffer
  moves the failure to the accumulator:
  `Fatal: Get different layout for acc` / `current layout: Fragment((16, 128) ->
  (8,), ... thread: 256 ...` / `previous layout: Fragment((16, 128) -> (16,),
  ... thread: 128, thread_range: I.Range(128, 256))`. Hoisting the branch to
  inside the loop body (one `T.gemm`, two predicated `T.copy`s) compiles but
  destroys the pipeline: 27.4 us against 12.9 us.
  Two `T.gemm`s in the *same* loop body over the *same* accumulator are fine --
  that is what the f32 weight split uses.
* `T.gemm` with `BN=16`: `Check failed: (m_warp * n_warp == num_warps) is false:
  m_warp * n_warp must equal num_warps, m_warp: 1, n_warp: 1, num_warps: 4`.
  BN=32 is the narrowest tile that works at 128 threads, which is what stops
  `w_out` from being tiled to 128 blocks over `N` and forces the K-split.
* **A dtype name used only in a `T.Tensor(...)` annotation.**
  `NameError: name 'cwdt' is not defined. Did you mean: '_wdt'?` -- the same
  closure-cell trap `basic.embed_row` documents for dimensions, and the reason
  `_f32` below takes the dtype as an argument.
* **Accumulating into a plain local.** `sb = sb + PB[h, e]` inside a loop, read
  after it: `RuntimeError: Immutable variable 'sb' is used outside its defining
  region!`. The eager builder makes every plain assignment an SSA binding scoped
  to its block; `T.alloc_var` plus `+=` is the mutable form. A Python `range`
  does not help -- it becomes a device loop like `T.serial`.
* **Shared memory is a hard wall, not a hint.** One stage too many gives
  `InternalError: Failed to set the allowed dynamic shared memory size to
  262144`, so every `stages` below is chosen against the 227 KB an H200 SM will
  hand out, and the f32 weight-split variants get one stage fewer to pay for
  their second tile.
* **Fragment extents that are not nice.** `NBA = 48` makes `KSL = 42` and the
  `(42, 32)` accumulator gives `InternalError: Check failed:
  (min_reg_num < (9223372036854775807L)) is false: no available layout found`.
"""
from __future__ import annotations

import functools
import math

import tilelang
import tilelang.language as T
import torch

try:  # `python -m kernels.gdn` and `import kernels.gdn` both have to work
    from .basic import BM
except ImportError:  # pragma: no cover -- `python kernels/gdn.py`
    from basic import BM

# ---------------------------------------------------------------------------
# `config.REAL`, spelled out. Every one of these appears in a `T.Tensor`
# annotation, and tilelang resolves annotation names against the kernel's
# *globals and closure cells* -- module level is the one scope that always
# resolves, so dimensions live here rather than as factory arguments (a factory
# argument that appears only in an annotation raises `NameError`; see
# `basic.embed_row`).
# ---------------------------------------------------------------------------
H = 2048  #: hidden
CONV = 8192  #: gdn_conv_dim = 2 * key + value
KEY = 2048  #: gdn_key_dim = 16 * 128
VAL = 4096  #: gdn_value_dim = 32 * 128
HK = 16  #: gdn_n_k_heads
HV = 32  #: gdn_n_v_heads
DK = 128  #: gdn_head_k_dim
DV = 128  #: gdn_head_v_dim
VPK = 2  #: value heads per key head
KERNEL = 4  #: gdn_conv_kernel
WINDOW = 3  #: gdn_conv_context -- the columns a step inherits

QSCALE = 1.0 / math.sqrt(DK)
L2_EPS = 1e-6  #: rsqrt of the *sum* of squares plus this, not the mean
RMS_EPS = 1e-6

#: K-splits of the (b, a) projections in kernel 1. Reading `w_in_b[:, h]` down a
#: column is stride-64 B and pulls a 32 B sector per useful 2 B -- 32 such blocks
#: cost 4.0 us. Splitting K instead keeps every read contiguous across all 32
#: heads at once, at the price of `NBA` partial sums that kernel 2 adds up.
#: Fewer, fatter splits win: 8 / 16 / 32 measured 12.0 / 12.9 / 13.0 us for the
#: whole of kernel 1, against 11.2 us with the (b, a) blocks removed entirely.
NBA = 8
KSL = H // NBA  #: 256 rows of K per (b, a) block

#: Value columns per delta-rule block, and the resulting groups per head.
NJ = 4
BJ = DV // NJ  #: 32

_TL_DTYPE = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


def _wdt(t: torch.Tensor) -> str:
    """The tilelang dtype name of a weight tensor, as a kernel-cache key."""
    return _TL_DTYPE[t.dtype]


def _f32(x, dt: str):
    """A weight element as f32; the cast is skipped when it already is one.

    Every kernel body below calls this with its own `wdt`/`cwdt`, and that is
    load-bearing beyond the cast: tilelang resolves the dtype name in the
    `T.Tensor(...)` annotations against the kernel function's **closure cells**,
    and Python creates a cell only for a name the nested function's code
    references. A dtype that appears only in an annotation raises
    `NameError: name 'cwdt' is not defined` -- the same trap `basic.embed_row`
    documents for dimensions.
    """
    return x if dt == "float32" else T.cast(x, "float32")


# ---------------------------------------------------------------------------
# 1. rms_norm(hidden) + in_proj_qkv + in_proj_b + in_proj_a
# ---------------------------------------------------------------------------

NQ = CONV // 128  #: 64 output tiles of the qkv projection


@functools.lru_cache(maxsize=None)
def _in_proj(wdt: str):
    """`entry`, and the raw (b, a) partial sums, from `hidden` and `gamma_in`.

    The layernorm is not a separate kernel: every block needs all of
    `hidden_norm`, so every block computes it. That is 16 KB of redundant L2
    traffic per block against 512 KB of weight tile -- 3%, and it saves a launch
    plus a round trip of 8 KB through HBM.

    `Zacc` and `Oacc` are zeroed here because kernels 2 and 3 accumulate into
    them with `T.atomic_add`, and a `torch.zeros` for each would be two more
    launches (measured 1.2 us apiece, against the 6.8 us `_out_proj` itself
    costs). A kernel boundary is the only grid-wide barrier available and this is
    the kernel before both, so 96 of the 8192 stores each qkv block already makes
    are spent on somebody else's output.
    """
    BN, BK, threads = 128, 128, 128
    KO = H // BK
    ZPB = VAL // NQ  #: z scalars each qkv block zeroes
    OPB = H // NQ  #: out scalars each qkv block zeroes
    # The f32 variant stages a second (low) weight tile, so it pays for it in
    # pipeline depth: 64 KB vector + stages * 32 KB + 32 KB low + 8 KB epilogue
    # has to stay under the 227 KB an SM will hand out.
    stages = 3 if wdt == "float32" else 4

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Hid: T.Tensor((H,), "float32"),
            Gin: T.Tensor((H,), "float32"),
            Wqkv: T.Tensor((H, CONV), wdt),
            Wb: T.Tensor((H, HV), wdt),
            Wa: T.Tensor((H, HV), wdt),
            Entry: T.Tensor((CONV,), "float32"),
            PB: T.Tensor((HV, NBA), "float32"),
            PA: T.Tensor((HV, NBA), "float32"),
            Zacc: T.Tensor((VAL,), "float32"),
            Oacc: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(NQ + NBA, threads=threads) as b:
                xs = T.alloc_shared((BM, H), "bfloat16")
                whi = T.alloc_shared((BK, BN), "bfloat16")
                wlo = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                sq = T.alloc_fragment((H,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                sc = T.alloc_shared((1,), "float32")
                pb = T.alloc_fragment((KSL, HV), "float32")
                pa = T.alloc_fragment((KSL, HV), "float32")
                rb = T.alloc_fragment((HV,), "float32")
                ra = T.alloc_fragment((HV,), "float32")

                for i in T.Parallel(H):
                    sq[i] = Hid[i] * Hid[i]
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    sc[0] = T.rsqrt(tot[0] / T.cast(H, "float32") + RMS_EPS)
                T.sync_threads()

                if b < NQ:
                    for i in T.Parallel(H):
                        # Row 0 is `hidden_norm` rounded to bf16, row 1 is what
                        # the rounding threw away. The MMA computes both rows
                        # against the same weight tile for free; the epilogue
                        # adds them back together.
                        xs[0, i] = T.cast(Hid[i] * sc[0] * Gin[i], "bfloat16")
                        xs[1, i] = T.cast(
                            Hid[i] * sc[0] * Gin[i]
                            - T.cast(T.cast(Hid[i] * sc[0] * Gin[i], "bfloat16"), "float32"),
                            "bfloat16",
                        )
                    for j in T.Parallel(ZPB):
                        Zacc[b * ZPB + j] = 0.0
                    for j in T.Parallel(OPB):
                        Oacc[b * OPB + j] = 0.0
                    T.clear(acc)
                    T.sync_threads()
                    for ko in T.Pipelined(KO, num_stages=stages):
                        T.copy(Wqkv[ko * BK:(ko + 1) * BK, b * BN:(b + 1) * BN], whi)
                        # `wdt` is compared here rather than in the factory on
                        # purpose: a name used only inside an annotation gets no
                        # closure cell and tilelang's evaluator then cannot see
                        # it. A Python-level `if` in the body (the same shape as
                        # `basic.gemv`'s `act`) both reads the name and folds
                        # away at trace time.
                        if wdt == "float32":
                            for u, v in T.Parallel(BK, BN):
                                wlo[u, v] = T.cast(
                                    T.cast(Wqkv[ko * BK + u, b * BN + v], "float32")
                                    - T.cast(whi[u, v], "float32"),
                                    "bfloat16",
                                )
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], whi, acc)
                        if wdt == "float32":
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], wlo, acc)
                    T.copy(acc, out)
                    for j in T.Parallel(BN):
                        Entry[b * BN + j] = out[0, j] + out[1, j]
                else:
                    # (b, a): 32 + 32 outputs over K=2048. Too narrow for a
                    # `T.gemm` tile and too small to be worth a launch, so it
                    # rides along in the 52 blocks kernel 1 leaves idle on 132
                    # SMs -- one K-slice each, all 32 heads at once, so every
                    # global read is 32 contiguous weights.
                    e = b - NQ
                    for m, hh in T.Parallel(KSL, HV):
                        pb[m, hh] = (
                            Hid[e * KSL + m] * sc[0] * Gin[e * KSL + m]
                            * T.cast(Wb[e * KSL + m, hh], "float32")
                        )
                        pa[m, hh] = (
                            Hid[e * KSL + m] * sc[0] * Gin[e * KSL + m]
                            * T.cast(Wa[e * KSL + m, hh], "float32")
                        )
                    T.reduce_sum(pb, rb, dim=0)
                    T.reduce_sum(pa, ra, dim=0)
                    for hh in T.Parallel(HV):
                        PB[hh, e] = rb[hh]
                        PA[hh, e] = ra[hh]

        return main

    return build()


# ---------------------------------------------------------------------------
# 2. in_proj_z + causal conv + l2 norm + the gated delta rule
# ---------------------------------------------------------------------------

NZ = VAL // 128  #: 32 output tiles of the z projection
ZKS = (HV * NJ) // NZ  #: 4 K-splits, so that one grid serves both roles


@functools.lru_cache(maxsize=None)
def _conv_delta(wdt: str, cwdt: str):
    """One grid, two independent jobs, because both want exactly 128 blocks.

    As a delta-rule block, `b` is `(value head, 32-column group)`. As a z block,
    the same `b` is `(128-wide output tile, K-quarter)`; `z` is split over K and
    accumulated with `T.atomic_add` rather than given a 32-wide tile per block
    (which would also be 128 blocks, and would need no atomics) because a
    128-wide tile pulls 256 B weight rows against a 32-wide one's 64 B. Out of
    HBM the two measure the same, 6.86 us against 6.87 us; out of L2 the wide
    tile wins by 0.9 us, and the wide tile also keeps the staged vector to
    `KC = 512` instead of all 2048, which is what leaves room for 5 pipeline
    stages. The two decompositions of `b` are unrelated and both cover their
    space exactly.

    The state tile is loaded into registers *before* the z pipeline runs. Those
    loads have no prefetch of their own, and issuing them early lets the gemm's
    ~4 us of pipeline hide their latency.
    """
    BN, BK, threads = 128, 128, 128
    KC = H // ZKS  #: 512 rows of K per z block
    KO = KC // BK
    stages = 4 if wdt == "float32" else 5

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Hid: T.Tensor((H,), "float32"),
            Gin: T.Tensor((H,), "float32"),
            Wz: T.Tensor((H, VAL), wdt),
            ConvS: T.Tensor((1, CONV, WINDOW), "float32"),
            Entry: T.Tensor((CONV,), "float32"),
            ConvW: T.Tensor((CONV, KERNEL), cwdt),
            Alog: T.Tensor((HV,), "float32"),
            Dtb: T.Tensor((HV,), "float32"),
            PB: T.Tensor((HV, NBA), "float32"),
            PA: T.Tensor((HV, NBA), "float32"),
            State: T.Tensor((1, HV, DK, DV), "float32"),
            Zacc: T.Tensor((VAL,), "float32"),
            Read: T.Tensor((1, HV, DV), "float32"),
            Up: T.Tensor((1, HV, DK, DV), "float32"),
        ):
            with T.Kernel(HV * NJ, threads=threads) as b:
                zn = b % NZ  # z output tile
                zk = b // NZ  # z K-quarter
                h = b // NJ  # value head
                j0 = (b % NJ) * BJ  # first value column
                hk = h // VPK  # the key head this value head shares

                xs = T.alloc_shared((BM, KC), "bfloat16")
                whi = T.alloc_shared((BK, BN), "bfloat16")
                wlo = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                sq = T.alloc_fragment((H,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                sc = T.alloc_shared((1,), "float32")
                dec = T.alloc_fragment((DK, BJ), "float32")
                tmp = T.alloc_fragment((DK, BJ), "float32")
                red = T.alloc_fragment((BJ,), "float32")
                csq = T.alloc_fragment((2, DK), "float32")
                ctot = T.alloc_fragment((2,), "float32")
                # Row 0 is q, row 1 is k -- see the convolution below.
                qk = T.alloc_shared((2, DK), "float32")
                vf = T.alloc_shared((BJ,), "float32")
                dl = T.alloc_shared((BJ,), "float32")
                nrm = T.alloc_shared((2,), "float32")
                eg = T.alloc_shared((1,), "float32")
                bta = T.alloc_shared((1,), "float32")

                for i in T.Parallel(H):
                    sq[i] = Hid[i] * Hid[i]
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    sc[0] = T.rsqrt(tot[0] / T.cast(H, "float32") + RMS_EPS)
                T.sync_threads()

                # Recurrent state first: 128 independent 4 B loads per thread
                # with nothing to hide them behind unless the gemm below does.
                for i, j in T.Parallel(DK, BJ):
                    dec[i, j] = State[0, h, i, j0 + j]

                for i in T.Parallel(KC):
                    xs[0, i] = T.cast(Hid[zk * KC + i] * sc[0] * Gin[zk * KC + i], "bfloat16")
                    xs[1, i] = T.cast(
                        Hid[zk * KC + i] * sc[0] * Gin[zk * KC + i]
                        - T.cast(
                            T.cast(Hid[zk * KC + i] * sc[0] * Gin[zk * KC + i], "bfloat16"),
                            "float32",
                        ),
                        "bfloat16",
                    )
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(KO, num_stages=stages):
                    T.copy(
                        Wz[zk * KC + ko * BK:zk * KC + (ko + 1) * BK, zn * BN:(zn + 1) * BN],
                        whi,
                    )
                    if wdt == "float32":
                        for u, v in T.Parallel(BK, BN):
                            wlo[u, v] = T.cast(
                                T.cast(Wz[zk * KC + ko * BK + u, zn * BN + v], "float32")
                                - T.cast(whi[u, v], "float32"),
                                "bfloat16",
                            )
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], whi, acc)
                    if wdt == "float32":
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], wlo, acc)
                T.copy(acc, out)
                for j in T.Parallel(BN):
                    T.atomic_add(Zacc[zn * BN + j], out[0, j] + out[1, j])

                # ---- the depthwise causal convolution, on this block's 288
                # channels only. `window = concat(conv_state, entry)` closes on
                # this token, so it is one dot product of length 4 per channel
                # and no channels mix. q and k are recomputed by all four blocks
                # of a head (and by both value heads of a key head): 8x
                # redundancy on 590 KB, which is free next to 16.8 MB of z.
                # q and k together: their channels are `hk * DK` and
                # `KEY + hk * DK`, so `t * KEY` selects between them and one
                # (2, DK) fragment reduced along `dim=1` gives both L2 norms in
                # a single allreduce instead of two (1.0 us off kernel 3 when
                # the same change was made to its four head norms).
                for t, i in T.Parallel(2, DK):
                    qk[t, i] = (
                        ConvS[0, t * KEY + hk * DK + i, 0]
                        * _f32(ConvW[t * KEY + hk * DK + i, 0], cwdt)
                        + ConvS[0, t * KEY + hk * DK + i, 1]
                        * _f32(ConvW[t * KEY + hk * DK + i, 1], cwdt)
                        + ConvS[0, t * KEY + hk * DK + i, 2]
                        * _f32(ConvW[t * KEY + hk * DK + i, 2], cwdt)
                        + Entry[t * KEY + hk * DK + i]
                        * _f32(ConvW[t * KEY + hk * DK + i, 3], cwdt)
                    )
                    qk[t, i] = qk[t, i] / (1.0 + T.exp(-qk[t, i]))  # silu
                    csq[t, i] = qk[t, i] * qk[t, i]
                T.reduce_sum(csq, ctot, dim=1)
                for t in T.Parallel(2):
                    nrm[t] = T.rsqrt(ctot[t] + L2_EPS)  # sum, not mean: `l2norm`
                T.sync_threads()
                for t, i in T.Parallel(2, DK):
                    qk[t, i] = qk[t, i] * nrm[t]

                for jj in T.Parallel(BJ):
                    vf[jj] = (
                        ConvS[0, 2 * KEY + h * DV + j0 + jj, 0]
                        * _f32(ConvW[2 * KEY + h * DV + j0 + jj, 0], cwdt)
                        + ConvS[0, 2 * KEY + h * DV + j0 + jj, 1]
                        * _f32(ConvW[2 * KEY + h * DV + j0 + jj, 1], cwdt)
                        + ConvS[0, 2 * KEY + h * DV + j0 + jj, 2]
                        * _f32(ConvW[2 * KEY + h * DV + j0 + jj, 2], cwdt)
                        + Entry[2 * KEY + h * DV + j0 + jj]
                        * _f32(ConvW[2 * KEY + h * DV + j0 + jj, 3], cwdt)
                    )
                    vf[jj] = vf[jj] / (1.0 + T.exp(-vf[jj]))

                # beta and g, from kernel 1's `NBA` partial sums. One thread:
                # 8 f32 to add, and every other thread would compute the same.
                #
                # `T.alloc_var` and `+=` -- not `sb = sb + PB[h, e]`, and not a
                # Python `range`. In the eager builder a plain assignment is an
                # *immutable* SSA binding scoped to the block it appears in, so
                # accumulating that way and reading the result after the loop
                # gives `RuntimeError: Immutable variable 'sb' is used outside
                # its defining region!`. `T.alloc_var` is the mutable one.
                if T.get_thread_binding() == 0:
                    sb = T.alloc_var("float32")
                    sa = T.alloc_var("float32")
                    sb = 0.0
                    sa = 0.0
                    for e in T.serial(NBA):
                        sb += PB[h, e]
                        sa += PA[h, e]
                    bta[0] = 1.0 / (1.0 + T.exp(-sb))
                    # softplus in the form that does not overflow: log1p of the
                    # negative branch plus the hinge. g <= 0 by construction, so
                    # exp(g) is a decay in (0, 1].
                    sa += Dtb[h]
                    eg[0] = T.exp(
                        -T.exp(Alog[h]) * (T.log(1.0 + T.exp(-T.abs(sa))) + T.max(sa, 0.0))
                    )
                T.sync_threads()

                # ---- the rank-one update, per value column.
                for i, j in T.Parallel(DK, BJ):
                    tmp[i, j] = dec[i, j] * eg[0] * qk[1, i]
                T.reduce_sum(tmp, red, dim=0)  # recalled[j], over the key axis
                for j in T.Parallel(BJ):
                    dl[j] = (vf[j] - red[j]) * bta[0]
                T.sync_threads()
                for i, j in T.Parallel(DK, BJ):
                    # `updated` is formed once and both consumed and stored, so
                    # `read` is literally `sum_i updated[i,j] * q[i]` rather than
                    # the algebraically equal `eg * (S.q) + delta * (k.q)` --
                    # which was also 0.9 us slower for needing a second pass.
                    Up[0, h, i, j0 + j] = dec[i, j] * eg[0] + qk[1, i] * dl[j]
                    tmp[i, j] = (dec[i, j] * eg[0] + qk[1, i] * dl[j]) * qk[0, i]
                T.reduce_sum(tmp, red, dim=0)
                for j in T.Parallel(BJ):
                    # `qscale` is applied to the 32 sums rather than to the 128
                    # q entries: same product, a quarter of the multiplies.
                    Read[0, h, j0 + j] = red[j] * QSCALE

        return main

    return build()


# ---------------------------------------------------------------------------
# 3. gated output norm + out_proj
# ---------------------------------------------------------------------------

NT = H // 128  #: 16 output tiles
OKS = 128 // NT  #: 8 K-splits -> 128 blocks


@functools.lru_cache(maxsize=None)
def _out_proj(wdt: str):
    """`out = (rms_norm(read, gamma) * silu(z)) @ w_out`.

    K=4096, N=2048 is 16 tiles of 128, i.e. 16 SMs and 1.26 TB/s. Splitting K
    eight ways and finishing with `T.atomic_add` puts 128 blocks on it: 6.8 us
    for the same 16.8 MB against 15.4 us. `KS=4` (64 blocks) was 7.0 us and
    `KS=16` (256) was 8.6 us -- past 128 the K loop is 2 iterations and the
    pipeline is all prologue.

    The K-split is what makes the gated norm fusible here at all. `KC = 512` is
    4 * `DV`, so a block's K-slice is four *whole* value heads and it can take
    the RMS over each head's 128 dims by itself.
    """
    BN, BK, threads = 128, 128, 128
    KC = VAL // OKS  #: 512
    KO = KC // BK
    HPB = KC // DV  #: 4 whole value heads per block
    stages = 4 if wdt == "float32" else 5

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Read: T.Tensor((1, HV, DV), "float32"),
            Zacc: T.Tensor((VAL,), "float32"),
            Ggdn: T.Tensor((DV,), "float32"),
            Wout: T.Tensor((VAL, H), wdt),
            Out: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(NT * OKS, threads=threads) as b:
                nt = b % NT
                ks = b // NT
                h0 = ks * HPB

                xs = T.alloc_shared((BM, KC), "bfloat16")
                whi = T.alloc_shared((BK, BN), "bfloat16")
                wlo = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                rsq = T.alloc_fragment((HPB, DV), "float32")
                rsm = T.alloc_fragment((HPB,), "float32")
                rsc = T.alloc_shared((HPB,), "float32")

                # All four head norms in one cross-thread reduction rather than
                # four in a `T.serial`: an allreduce over 128 threads costs
                # ~0.2 us in barriers and the reduce is over `DV`, which is the
                # fragment's *inner* axis, so `dim=1` needs no shuffles at all.
                for hh, j in T.Parallel(HPB, DV):
                    rsq[hh, j] = Read[0, h0 + hh, j] * Read[0, h0 + hh, j]
                T.reduce_sum(rsq, rsm, dim=1)
                for hh in T.Parallel(HPB):
                    # `Qwen3_5MoeRMSNormGated` is flat (`weight * x`, no `1 +`)
                    # and normalises before the gate.
                    rsc[hh] = T.rsqrt(rsm[hh] / T.cast(DV, "float32") + RMS_EPS)
                T.sync_threads()

                for hh, j in T.Parallel(HPB, DV):
                    xs[0, hh * DV + j] = T.cast(
                        Read[0, h0 + hh, j] * rsc[hh] * Ggdn[j]
                        * (
                            Zacc[(h0 + hh) * DV + j]
                            / (1.0 + T.exp(-Zacc[(h0 + hh) * DV + j]))
                        ),
                        "bfloat16",
                    )
                    xs[1, hh * DV + j] = T.cast(
                        Read[0, h0 + hh, j] * rsc[hh] * Ggdn[j]
                        * (
                            Zacc[(h0 + hh) * DV + j]
                            / (1.0 + T.exp(-Zacc[(h0 + hh) * DV + j]))
                        )
                        - T.cast(xs[0, hh * DV + j], "float32"),
                        "bfloat16",
                    )
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(KO, num_stages=stages):
                    T.copy(
                        Wout[ks * KC + ko * BK:ks * KC + (ko + 1) * BK, nt * BN:(nt + 1) * BN],
                        whi,
                    )
                    if wdt == "float32":
                        for u, v in T.Parallel(BK, BN):
                            wlo[u, v] = T.cast(
                                T.cast(Wout[ks * KC + ko * BK + u, nt * BN + v], "float32")
                                - T.cast(whi[u, v], "float32"),
                                "bfloat16",
                            )
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], whi, acc)
                    if wdt == "float32":
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], wlo, acc)
                T.copy(acc, out)
                for j in T.Parallel(BN):
                    T.atomic_add(Out[nt * BN + j], out[0, j] + out[1, j])

        return main

    return build()


# ---------------------------------------------------------------------------
# The three standalone `@func` boundaries. Selectable through `TF_IMPL`, so a
# wrong output bisects to one of them instead of to the fused step.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _conv_step(cwdt: str):
    """`silu(sum_j window[:, :, j] * conv_w[:, j])` over all 8192 channels.

    288 KB of traffic and no reduction longer than 4, so this is entirely a
    launch: one thread per channel, 32 blocks, and the four weights of a channel
    are 16 contiguous bytes so the load vectorises.
    """
    threads = 256

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            ConvS: T.Tensor((1, CONV, WINDOW), "float32"),
            Entry: T.Tensor((1, CONV, 1), "float32"),
            ConvW: T.Tensor((CONV, KERNEL), cwdt),
            Y: T.Tensor((1, CONV), "float32"),
        ):
            with T.Kernel(T.ceildiv(CONV, threads), threads=threads) as b:
                for t in T.Parallel(threads):
                    c = b * threads + t
                    Y[0, c] = (
                        ConvS[0, c, 0] * _f32(ConvW[c, 0], cwdt)
                        + ConvS[0, c, 1] * _f32(ConvW[c, 1], cwdt)
                        + ConvS[0, c, 2] * _f32(ConvW[c, 2], cwdt)
                        + Entry[0, c, 0] * _f32(ConvW[c, 3], cwdt)
                    )
                    Y[0, c] = Y[0, c] / (1.0 + T.exp(-Y[0, c]))

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _l2_normalise():
    """`x * rsqrt(sum(x^2) + 1e-6)` per head. One block per head, 128 threads."""

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, 1, HV, DK), "float32"),
            Y: T.Tensor((1, 1, HV, DK), "float32"),
        ):
            with T.Kernel(HV, threads=DK) as h:
                sq = T.alloc_fragment((DK,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                sc = T.alloc_shared((1,), "float32")
                for i in T.Parallel(DK):
                    sq[i] = X[0, 0, h, i] * X[0, 0, h, i]
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    sc[0] = T.rsqrt(tot[0] + L2_EPS)
                T.sync_threads()
                for i in T.Parallel(DK):
                    Y[0, 0, h, i] = X[0, 0, h, i] * sc[0]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _delta_step():
    """The delta rule alone, same `(head, column group)` grid as `_conv_delta`."""
    threads = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            State: T.Tensor((1, HV, DK, DV), "float32"),
            Q: T.Tensor((1, 1, HV, DK), "float32"),
            K: T.Tensor((1, 1, HV, DK), "float32"),
            V: T.Tensor((1, 1, HV, DV), "float32"),
            G: T.Tensor((1, 1, HV), "float32"),
            Bt: T.Tensor((1, 1, HV), "float32"),
            Read: T.Tensor((1, HV, DV), "float32"),
            Up: T.Tensor((1, HV, DK, DV), "float32"),
        ):
            with T.Kernel(HV * NJ, threads=threads) as b:
                h = b // NJ
                j0 = (b % NJ) * BJ
                dec = T.alloc_fragment((DK, BJ), "float32")
                tmp = T.alloc_fragment((DK, BJ), "float32")
                red = T.alloc_fragment((BJ,), "float32")
                qf = T.alloc_shared((DK,), "float32")
                kf = T.alloc_shared((DK,), "float32")
                dl = T.alloc_shared((BJ,), "float32")
                eg = T.alloc_shared((1,), "float32")
                bta = T.alloc_shared((1,), "float32")
                for i in T.Parallel(DK):
                    kf[i] = K[0, 0, h, i]
                    qf[i] = Q[0, 0, h, i] * QSCALE
                if T.get_thread_binding() == 0:
                    eg[0] = T.exp(G[0, 0, h])
                    bta[0] = Bt[0, 0, h]
                T.sync_threads()
                for i, j in T.Parallel(DK, BJ):
                    dec[i, j] = State[0, h, i, j0 + j] * eg[0]
                for i, j in T.Parallel(DK, BJ):
                    tmp[i, j] = dec[i, j] * kf[i]
                T.reduce_sum(tmp, red, dim=0)
                for j in T.Parallel(BJ):
                    dl[j] = (V[0, 0, h, j0 + j] - red[j]) * bta[0]
                T.sync_threads()
                for i, j in T.Parallel(DK, BJ):
                    Up[0, h, i, j0 + j] = dec[i, j] + kf[i] * dl[j]
                    tmp[i, j] = (dec[i, j] + kf[i] * dl[j]) * qf[i]
                T.reduce_sum(tmp, red, dim=0)
                for j in T.Parallel(BJ):
                    Read[0, h, j0 + j] = red[j]

        return main

    return build()


# ---------------------------------------------------------------------------
# Entry points. Each allocates its outputs and returns them; none synchronises,
# branches on a device value, or has a shape that depends on one, so all four
# capture into a CUDA graph.
# ---------------------------------------------------------------------------


def conv_step(conv_state: torch.Tensor, entry: torch.Tensor, conv_w: torch.Tensor):
    """`(1, CONV)` f32 -- the depthwise causal conv at one token, silu'd."""
    y = torch.empty((1, CONV), dtype=torch.float32, device=conv_state.device)
    _conv_step(_wdt(conv_w))(conv_state, entry, conv_w, y)
    return y


def l2_normalise(x: torch.Tensor):
    """`(1, 1, HV, DK)` f32 -- per-head L2 normalisation."""
    y = torch.empty_like(x)
    _l2_normalise()(x, y)
    return y


def delta_step(
    recurrent_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    """`(read (1, HV, DV), updated (1, HV, DK, DV))`, both f32.

    The state is an output because a rank-one update has no smaller increment to
    hand back, and it stays f32 throughout: the published config says
    `mamba_ssm_dtype: float32` and this error compounds over the whole sequence.
    """
    read = torch.empty((1, HV, DV), dtype=torch.float32, device=q.device)
    updated = torch.empty_like(recurrent_state)
    _delta_step()(recurrent_state, q, k, v, g, beta, read, updated)
    return read, updated


def linear_attention(
    hidden: torch.Tensor,
    gamma_in: torch.Tensor,
    w_in_qkv: torch.Tensor,
    w_in_z: torch.Tensor,
    w_in_b: torch.Tensor,
    w_in_a: torch.Tensor,
    conv_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    gamma_gdn: torch.Tensor,
    w_out: torch.Tensor,
):
    """`(out (1, 1, H), entry (1, CONV, 1), updated (1, HV, DK, DV))`.

    Three launches. `entry` is returned because the caller owns the convolution
    window and joins this step's column on; `conv_state` is read-only here.
    """
    # tilelang's packed ABI requires the declared row-major strides, and the
    # convolution window the caller hands back is a **sliced view**: the shipped
    # `advance_state` is `cat([window, column], dim=2)[:, :, -WINDOW:]`, whose
    # stride[1] is KERNEL (4), not WINDOW (3). Passing it straight through gives
    #
    #   RuntimeError: kernel main input ConvS strides[1] violates packed ABI
    #   constraint; expected: 4, got: 3
    #
    # on the second decode step and not the first, because `init_caches` starts
    # with a contiguous tensor. 98 KB per layer per token is 3 MB against this
    # step's 5.9 GB, so the copy is free; the alternative is a second compiled
    # variant per stride, which is a lot of machinery for a rounding error's
    # worth of bytes. Cheap and unconditional beats correct-only-sometimes.
    conv_state = conv_state.contiguous()
    dev = hidden.device
    kw = {"dtype": torch.float32, "device": dev}
    entry = torch.empty((1, CONV, 1), **kw)
    out = torch.empty((1, 1, H), **kw)
    updated = torch.empty((1, HV, DK, DV), **kw)
    read = torch.empty((1, HV, DV), **kw)
    # Cross-kernel scratch: the (b, a) partial sums, and the z accumulator that
    # kernel 2 atomically fills. Both are pool allocations, not launches.
    pb = torch.empty((HV, NBA), **kw)
    pa = torch.empty((HV, NBA), **kw)
    zacc = torch.empty((VAL,), **kw)

    _in_proj(_wdt(w_in_qkv))(
        hidden.view(H), gamma_in, w_in_qkv.view(H, CONV),
        w_in_b.view(H, HV), w_in_a.view(H, HV),
        entry.view(CONV), pb, pa, zacc, out.view(H),
    )
    _conv_delta(_wdt(w_in_z), _wdt(conv_w))(
        hidden.view(H), gamma_in, w_in_z.view(H, VAL),
        conv_state, entry.view(CONV), conv_w, a_log, dt_bias, pb, pa,
        recurrent_state, zacc, read, updated,
    )
    _out_proj(_wdt(w_out))(read, zacc, gamma_gdn, w_out.view(VAL, H), out.view(H))
    return out, entry, updated


__all__ = ["conv_step", "delta_step", "l2_normalise", "linear_attention"]


# ---------------------------------------------------------------------------
# Self-test: plain torch, the authored module, and the clock.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import time

    import torch.nn.functional as F

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    torch.manual_seed(0)
    DEV = "cuda"

    def rel(a, b):
        """max|a - b| / max|b|.

        Scale-relative, not elementwise: these vectors have entries that pass
        through zero (silu of a centred projection, a delta near convergence),
        and an elementwise ratio reports 1e0 for an absolute error of 1e-9.
        """
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()

    def graph_bench(call, reps=100, iters=20):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                call()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters / reps * 1e6

    print(f"device {torch.cuda.get_device_name()}  tilelang {tilelang.__version__}")

    # ---- random activations, and weights at both dtypes ------------------
    hidden = torch.randn(1, 1, H, device=DEV)
    conv_state = torch.randn(1, CONV, WINDOW, device=DEV)
    rstate = torch.randn(1, HV, DK, DV, device=DEV)
    # Scaled so activations stay O(1) through a K=2048 contraction, which is
    # what the real (converted) checkpoint looks like; a unit-variance weight
    # would make every intermediate ~45 and the tolerances meaningless.
    w32 = {
        "gamma_in": 1.0 + 0.02 * torch.randn(H, device=DEV),
        "w_in_qkv": torch.randn(1, H, CONV, device=DEV) / H**0.5,
        "w_in_z": torch.randn(1, H, VAL, device=DEV) / H**0.5,
        "w_in_b": torch.randn(1, H, HV, device=DEV) / H**0.5,
        "w_in_a": torch.randn(1, H, HV, device=DEV) / H**0.5,
        "conv_w": torch.randn(CONV, KERNEL, device=DEV) / KERNEL**0.5,
        "a_log": torch.rand(HV, device=DEV).mul(16).clamp_min(1e-3).log(),
        "dt_bias": torch.randn(HV, device=DEV),
        "gamma_gdn": 1.0 + 0.02 * torch.randn(DV, device=DEV),
        "w_out": torch.randn(1, VAL, H, device=DEV) / VAL**0.5,
    }
    wbf = {k: (v.bfloat16() if v.dim() >= 2 else v) for k, v in w32.items()}

    # ---- (a) plain torch, written out here ------------------------------
    def t_conv(cs, e, cw):
        window = torch.cat([cs, e], dim=2)
        return F.silu((window * cw.float().reshape(1, CONV, KERNEL)).sum(-1))

    def t_l2(x):
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)

    def t_delta(S, q, k, v, g, beta):
        decayed = S * g.exp().reshape(1, HV, 1, 1)
        kc = k.reshape(1, HV, DK, 1)
        recalled = (decayed * kc).sum(-2)
        delta = (v.reshape(1, HV, DV) - recalled) * beta.reshape(1, HV, 1)
        upd = decayed + kc * delta.reshape(1, HV, 1, DV)
        read = (upd * (q * QSCALE).reshape(1, HV, DK, 1)).sum(-2)
        return read, upd

    def t_linear(hid, cs, rs, w):
        hn = hid * torch.rsqrt(hid.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w["gamma_in"]
        ent = (hn @ w["w_in_qkv"].float()).reshape(1, CONV, 1)
        mixed = t_conv(cs, ent, w["conv_w"])
        qf, kf, vf = mixed[:, :KEY], mixed[:, KEY:2 * KEY], mixed[:, 2 * KEY:CONV]
        q = t_l2(qf.reshape(1, 1, HK, DK).repeat_interleave(VPK, dim=2))
        k = t_l2(kf.reshape(1, 1, HK, DK).repeat_interleave(VPK, dim=2))
        v = vf.reshape(1, 1, HV, DV)
        beta = torch.sigmoid(hn @ w["w_in_b"].float())
        g = -w["a_log"].exp() * F.softplus(hn @ w["w_in_a"].float() + w["dt_bias"])
        read, upd = t_delta(rs, q, k, v, g.reshape(1, 1, HV), beta.reshape(1, 1, HV))
        z = (hn @ w["w_in_z"].float()).reshape(1, HV, DV)
        normed = read * torch.rsqrt(read.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w["gamma_gdn"]
        gated = normed * F.silu(z)
        return gated.reshape(1, 1, VAL) @ w["w_out"].float(), ent, upd

    print("\n--- (a) against plain torch, written inline above -------------")
    y = conv_step(conv_state, (hidden @ w32["w_in_qkv"]).reshape(1, CONV, 1), w32["conv_w"])
    print(f"conv_step      mixed    {rel(y, t_conv(conv_state, (hidden @ w32['w_in_qkv']).reshape(1, CONV, 1), w32['conv_w'])):.2e}   (< 1e-5)")

    xin = torch.randn(1, 1, HV, DK, device=DEV)
    print(f"l2_normalise   y        {rel(l2_normalise(xin), t_l2(xin)):.2e}   (< 1e-5)")

    dq = torch.randn(1, 1, HV, DK, device=DEV)
    dk = torch.randn(1, 1, HV, DK, device=DEV)
    dv = torch.randn(1, 1, HV, DV, device=DEV)
    dg = -torch.rand(1, 1, HV, device=DEV) * 4.0
    db = torch.rand(1, 1, HV, device=DEV)
    kr, ku = delta_step(rstate, dq, dk, dv, dg, db)
    tr, tu = t_delta(rstate, dq, dk, dv, dg, db)
    print(f"delta_step     read     {rel(kr, tr):.2e}   (< 1e-5)")
    print(f"delta_step     updated  {rel(ku, tu):.2e}   (< 1e-5)")

    # Two references, because two different things are being asked.
    #  * `wbf` upcast: the torch reference on the *same weight values* the bf16
    #    kernel reads. This is the kernel's own arithmetic error, and it is what
    #    the < 2e-3 gate is about.
    #  * `w32`: the torch reference on the unrounded weights. Against the bf16
    #    kernel this measures bf16 *storage*, which no kernel can undo -- a
    #    single K=2048 projection is already 1.7e-3 there (the weight mantissa is
    #    8 bits, so the dot product inherits ~2^-9/sqrt(3) of relative error),
    #    and `out` is three such projections chained through a softplus.
    ref32 = t_linear(hidden, conv_state, rstate, w32)
    refbf = t_linear(hidden, conv_state, rstate, {k: v.float() for k, v in wbf.items()})
    for tag, w, ref, gate in (
        ("bf16 w vs same-w torch", wbf, refbf, " (< 2e-3)"),
        ("f32  w vs f32 torch   ", w32, ref32, " (< 2e-3)"),
        ("bf16 w vs f32 torch   ", wbf, ref32, " = bf16 storage"),
    ):
        o, e, u = linear_attention(
            hidden, w["gamma_in"], w["w_in_qkv"], w["w_in_z"], w["w_in_b"], w["w_in_a"],
            w["conv_w"], w["a_log"], w["dt_bias"], conv_state, rstate, w["gamma_gdn"],
            w["w_out"],
        )
        print(
            f"linear_attention {tag}  out {rel(o, ref[0]):.2e}"
            f"  entry {rel(e, ref[1]):.2e}  updated {rel(u, ref[2]):.2e}{gate}"
        )

    # ---- (b) against the authored module -------------------------------
    print("\n--- (b) against model.py:Qwen3_5LinearAttention (f32 weights) -")
    import model
    from tilefoundry.runtime import DictResource

    mod = model.Qwen3_5LinearAttention
    loaded = mod.load(DictResource({n: w32[n] for n in mod.weights}))

    a_out, a_ent, a_upd = loaded.linear_attention(hidden, conv_state, rstate)
    o, e, u = linear_attention(
        hidden, w32["gamma_in"], w32["w_in_qkv"], w32["w_in_z"], w32["w_in_b"],
        w32["w_in_a"], w32["conv_w"], w32["a_log"], w32["dt_bias"], conv_state, rstate,
        w32["gamma_gdn"], w32["w_out"],
    )
    print(f"linear_attention  out      {rel(o, a_out):.2e}")
    print(f"linear_attention  entry    {rel(e, a_ent):.2e}")
    print(f"linear_attention  updated  {rel(u, a_upd):.2e}")

    a_mixed = mod.conv_step(conv_state, a_ent, w32["conv_w"])
    print(f"conv_step         mixed    {rel(conv_step(conv_state, a_ent, w32['conv_w']), a_mixed):.2e}")
    print(f"l2_normalise      y        {rel(l2_normalise(xin), mod.l2_normalise(xin)):.2e}")
    b_read, b_upd = mod.delta_step(rstate, dq, dk, dv, dg, db)
    print(f"delta_step        read     {rel(kr, b_read):.2e}")
    print(f"delta_step        updated  {rel(ku, b_upd):.2e}")

    # Kernels 2 and 3 accumulate into `zacc` / `out` with `T.atomic_add` and
    # kernel 1 is what zeroes them. If that ever stopped happening the answer
    # would drift call over call rather than be wrong once, so: call it again.
    o2, e2, u2 = linear_attention(
        hidden, w32["gamma_in"], w32["w_in_qkv"], w32["w_in_z"], w32["w_in_b"],
        w32["w_in_a"], w32["conv_w"], w32["a_log"], w32["dt_bias"], conv_state, rstate,
        w32["gamma_gdn"], w32["w_out"],
    )
    print(f"repeat-invariance out      {rel(o2, o):.2e}   (atomics re-zeroed)")

    # ---- (c) the clock, inside a CUDA graph ----------------------------
    #
    # The harness replays the *same* call 100 times, so a working set that fits
    # in the 50 MB L2 would be measured out of L2 and not out of HBM.
    # `linear_attention` moves 67 MB of weights, which does not fit -- rotating
    # three separate weight sets through the same graph gave 32.87 us against
    # 32.64 us for one, so the number below is the HBM-bound one. The two small
    # kernels are launch-bound and their TB/s is meaningless; they are here to
    # show the floor a launch costs.
    print("\n--- (c) wall time per call, captured in a CUDA graph ----------")
    ent_col = (hidden @ w32["w_in_qkv"]).reshape(1, CONV, 1)
    for name, call, bytes_moved in (
        ("conv_step", lambda: conv_step(conv_state, ent_col, wbf["conv_w"]), 0),
        ("l2_normalise", lambda: l2_normalise(xin), 0),
        ("delta_step", lambda: delta_step(rstate, dq, dk, dv, dg, db), 4 << 20),
        (
            "linear_attention",
            lambda: linear_attention(
                hidden, wbf["gamma_in"], wbf["w_in_qkv"], wbf["w_in_z"], wbf["w_in_b"],
                wbf["w_in_a"], wbf["conv_w"], wbf["a_log"], wbf["dt_bias"], conv_state,
                rstate, wbf["gamma_gdn"], wbf["w_out"],
            ),
            (H * CONV + H * VAL + VAL * H) * 2 + (4 << 20),
        ),
    ):
        us = graph_bench(call)
        rate = f"{bytes_moved / us * 1e-6:5.2f} TB/s" if bytes_moved else "launch-bound"
        print(f"{name:17s} {us:7.2f} us   {rate}")
