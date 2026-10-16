# GDN mixer kernels (runtime/kernels/gdn.py) -- notes

Scope: the GatedDeltaNet token mixer of Qwen3.8-27B prefill between the fused input
projection and `w_out`: conv + silu + l2norm + gates (`gdn_prep`), chunked gated delta
rule (`gdn_chunk` + `gdn_scan`), gated RMSNorm (fused into `gdn_scan`).
tilelang 0.1.15, torch 2.14.1+cu130, H200 (GPU1, SM clock 1500 MHz), CUDA 13.2.

## Kernels

| kernel | grid x threads | does |
|---|---|---|
| `gdn_prep` | (80 head columns, T/(R*W)) x 32W, (R,W)=(2,4) for T<=256 else (4,4) | causal depthwise conv (f32 accumulate in tap order), silu, l2norm of q/k in bf16 arithmetic, v, g, beta, conv window. Lane = 4 channels, warp = R rows of one 128-channel head. silu: fast path for the lane's 4R values, one branch to an exact recomputation of the values near a bf16 rounding midpoint; outputs stored as 8-byte bf16 vectors |
| `gdn_chunk` | (T/64 chunks, 48 value heads) x 256 | per chunk c, head h: G = cumsum g; KK, QK (wgmma); A = strict_lower(beta_i e^{G_i-G_j} k_i.k_j); X = (I+A)^-1 (f32 SIMT, 16x16 diagonal blocks by forward substitution, then two doubling levels 16->32->64: X21 = -X22 (A21 X11), 1x2 / 2x2 register tiles, 5 phases); U^T = V^T (X diag beta)^T and W = (X diag(beta e^G)) K (wgmma, X side split in bf16 hi+lo); P = lower(q.k e^{G_i-G_j})/sqrt(128); decay factors F. Writes U^T (f32), W (bf16 hi and lo), P (bf16), F (f32) through swizzled smem + TMA stores |
| `gdn_scan_plain` (production) / `gdn_scan` (fused_norm=True) | (2 dv halves, 48) x 384; plain: independent CTAs (no cluster); fused: 2-CTA cluster | warp specialised. WG0 (math, 240 regs): S^T (64 dv x 128 dk, f32) stays in the wgmma accumulator; per chunk, with bf16 hi+lo splits S = Sh+Sl, W = Wh+Wl, D = Delta^T e^{G63-G} = Dh+Dl: Y = Sh Wh^T + Sh Wl^T + Sl Wh^T, Delta^T = U^T - Y (f32), O = (Sh Q^T) e^G/sqrt(128) + bf16(Delta^T) P^T, S^T = e^{G63} S^T + Dh K + Dl K. WG1: TMA producer (2-stage ring) + epilogue rows 0-31. WG2: epilogue rows 32-63. Plain variant (production): the epilogue stores bf16(o) (T, 48, 128). Fused variant (only with `GdnKernels(T, fused_norm=True)`, for timing): row sums of o^2 exchanged with the peer CTA (DSMEM bulk copy + remote mbarrier), gated RMSNorm with fast silu, y stores. The fused variant is not used in production because its 2-CTA exchange cannot be stated in TileFoundry scheduled HIR |
| `gdn_gated_norm` | T*48/8 CTAs x 128 (half-warp per (t, head) row, lane = 8 channels, 16-byte loads) | production gated RMSNorm of the bf16 read-out o (workspace `ws_o`): bitwise equal to model.py `Qwen3_5RMSNormGated` on the same o. Sum of squares in torch's order for `rf.square().mean(-1)` (ATen Reduce.cuh, input-vectorised case: 32 virtual lanes own 4-element chunks, ((x0^2+x1^2)+x2^2)+x3^2, shfl_down 16..1, * 1/128); IEEE ops without FMA, `rsqrtf`; gate n2*silu(z) by ex2/rcp approx for the lane's 8 values and one branch to the exact n2*z/(1+expf(-z)) for values near a bf16 rounding midpoint (margin 16 f32 ulps; checked over all 65280^2 finite bf16 (n2, z) pairs); compiled without fast-math |

Workspaces (allocated once per `GdnKernels`): U^T (48, NC, 128, 64) f32, W (48, NC, 2, 64, 128)
bf16 (hi, lo), P (48, NC, 64, 64) bf16, F (48, NC, 3, 64) f32, and the read-out o (T, 48, 128)
bf16 between the plain scan and the gated norm (not allocated with fused_norm=True);
70 MB at T=1024, 8.7 MB at T=128.

## Math (derived from the recurrence in the task, no outside source)

Per value head, chunk of 64 rows, S0 = state entering the chunk, G = inclusive cumsum of g:

    S_i  = e^{G_i} S0 + sum_{j<=i} e^{G_i-G_j} k_j delta_j^T
    delta_i = beta_i (v_i - (e^{G_i} S0 + sum_{j<i} e^{G_i-G_j} k_j delta_j^T)^T k_i)
    => (I + A) Delta = diag(beta) V - diag(beta e^G) K S0,  A_ij = beta_i e^{G_i-G_j} k_i.k_j (j<i)
    => Delta = U - W S0,  U = X diag(beta) V,  W = X diag(beta e^G) K,  X = (I + A)^-1
    O    = diag(e^G) Q~ S0 + P Delta,       P_ij = e^{G_i-G_j} q~_i.k_j (j<=i)
    S_63 = e^{G_63} S0 + (K e^{G_63-G})^T Delta

All decays are exp of non-positive numbers (g <= 0); no exp(+G) factorisation is used.
The scan runs on S^T so that S^T and Delta^T are wgmma A operands straight from registers
(the accumulator layout of m64nNk16 equals the A-fragment layout), no shared-memory round
trip of the state.

## Precision choices

* Tensor-core operands are bf16 with f32 accumulation; the recurrent state accumulates in
  f32 in the wgmma accumulator.
* Read-out path (O = q.S + P.Delta): single bf16 operands (y error budget allows it).
* State path, first version (all single bf16 operands, U^T/W stored in bf16): state
  rel_l2 1.5e-3..2.1e-3, but per element up to 1.6% relative error on large elements
  (T=128 layer0: max |d| 6.5e-2 at |ref| ~3); a per-element allclose (rtol 0.009,
  atol 0.009*RMS) failed by up to 9.7x. A torch emulation of the dataflow, switching one
  rounding at a time, showed that four roundings contribute and each alone fixes only part:
  S -> bf16 in W.S, W stored in bf16, U stored in bf16, and Delta -> bf16 in the K^T.Delta
  update (worst |d|/tol, T=128 layer0: all bf16 9.7; U f32 5.0; Delta split 5.9; S split 8.9;
  U f32 + S split + Delta split with W bf16 2.4; all four fixed 0.022; tf32 everywhere 0.54).
* Current (default): U^T in f32; W stored as bf16 hi + lo; W.S as Sh.Wh + Sh.Wl + Sl.Wh;
  state update as Dh.K + Dl.K (K is exact in bf16); X diag(beta) and X diag(beta e^G) split
  hi + lo in the chunk kernel. Per element worst |d|/tol <= 0.032 at every T and layer tested.

Fast (first version) against precise (current default), state vs HIR f32 reference;
maxrel = max |d|/|ref| over elements with |ref| > 10% of max |ref|; chunk+scan time is
amortised (CUDA graph of 20 back-to-back `core()` calls, median of 10 replays):

| T | layer | version | state max abs | maxrel | state rel_l2 | worst abs/tol | y rel_l2 | chunk+scan us |
|---|---|---|---|---|---|---|---|---|
| 128 | 0 | fast | 6.49e-2 | 1.56e-2 | 1.50e-3 | 9.31 (fails) | 1.145e-3 | 20.1 |
| 128 | 0 | precise | 3.34e-4 | 2.70e-5 | 4.53e-6 | 0.028 | 1.140e-3 | 22.0 |
| 128 | 60 | fast | 7.73e-3 | 1.04e-2 | 2.15e-3 | 6.14 (fails) | 4.350e-3 | 20.1 |
| 128 | 60 | precise | 1.30e-5 | 1.22e-5 | 4.53e-6 | 0.023 | 4.090e-3 | 22.3 |
| 1024 | 0 | fast | 7.76e-2 | 7.24e-3 | 1.62e-3 | 6.62 (fails) | 9.202e-4 | 84.1 |
| 1024 | 0 | precise | 6.73e-4 | 3.33e-5 | 6.00e-6 | 0.032 | 8.615e-4 | 95.1 |
| 1024 | 60 | fast | 7.60e-3 | 5.70e-3 | 1.79e-3 | 5.01 (fails) | 4.574e-3 | 84.0 |
| 1024 | 60 | precise | 2.34e-5 | 3.74e-5 | 4.77e-6 | 0.024 | 4.105e-3 | 96.1 |

Cost of the precise state path: about +2 us (+10%) at T=128 and +11.5 us (+14%) at T=1024
for chunk+scan. The fast version fails the per-element check, so the precise one is the
default (there is no switch; the fast version is not kept in gdn.py).

## Correctness (test_gdn.py, layers 0 and 60, realistic inputs, proj buffer stride 16640)

prep outputs q, k, v, beta, window, g: **bitwise equal** to the HIR at all four T, both
layers (0 mismatching elements; g also bitwise). Exhaustive check: every finite bf16 value
(65280) as a conv output (taps 0,0,0,1) -> silu path bitwise equal to torch.

| T | layer | y rel_l2 | y cos | state rel_l2 | state max abs | state worst abs/tol | read (z=None) rel_l2 | key_heads=48 read / state rel_l2 |
|---|---|---|---|---|---|---|---|---|
| 128 | 0 | 1.139e-3 | 0.9999994 | 4.55e-6 | 3.57e-4 | 0.027 | 1.750e-3 | 3.10e-3 / 3.3e-6 |
| 128 | 60 | 4.090e-3 | 0.9999917 | 4.55e-6 | 1.35e-5 | 0.024 | 2.750e-3 | 3.25e-3 / 3.3e-6 |
| 256 | 0 | 1.054e-3 | 0.9999994 | 6.42e-6 | 3.15e-4 | 0.029 | 1.649e-3 | 3.11e-3 / 3.7e-6 |
| 256 | 60 | 4.054e-3 | 0.9999918 | 3.59e-6 | 1.25e-5 | 0.021 | 2.445e-3 | 3.03e-3 / 3.5e-6 |
| 512 | 0 | 9.440e-4 | 0.9999996 | 5.54e-6 | 4.27e-4 | 0.026 | 1.648e-3 | 2.96e-3 / 3.9e-6 |
| 512 | 60 | 3.934e-3 | 0.9999923 | 5.31e-6 | 2.15e-5 | 0.020 | 2.269e-3 | 2.93e-3 / 3.7e-6 |
| 1024 | 0 | 8.615e-4 | 0.9999996 | 6.13e-6 | 7.36e-4 | 0.032 | 1.644e-3 | 3.03e-3 / 4.2e-6 |
| 1024 | 60 | 4.105e-3 | 0.9999916 | 4.73e-6 | 2.38e-5 | 0.024 | 2.159e-3 | 2.88e-3 / 4.1e-6 |

(Numbers after the SOL pass; the new inverse order of `gdn_chunk` changes state bits at the
1e-6 level; before it: state rel_l2 4.53e-6..6.48e-6, worst abs/tol 0.020..0.032.)

"worst abs/tol" is max |d| / (0.009 |ref| + 0.009 RMS(ref)) (per-element allclose; < 1 passes).
Targets: y rel_l2 <= 0.009, cos >= 0.99996, state rel_l2 <= 0.009 and per element: all met.
The y error is dominated by the delta-rule read-out (bf16 operands). The production gated
norm (`gdn_gated_norm`) applied to the kernel's own read is bitwise equal to the HIR formula
(0 differing elements at all T, layers 0 and 60; also 0 on random rows with magnitudes over
many octaves). The old fused norm (fused_norm=True) differs in 2..30 of 0.8M..6.3M elements
by 1 bf16 ulp (fast `ex2/rcp` silu of z). The final state is bitwise the same on all paths
(z given, z=None, fused_norm=True).
CUDA graph capture (side stream) of prep + core replays bitwise equal to eager.

## Timing

Method: each piece captured alone in a CUDA graph; events around `graph.replay()` with a
GPU sleep queued before, median of 50 replays after 5 warm-up replays. A single-kernel
graph carries ~3-4 us of graph-launch latency, so per-kernel lines overstate the in-graph
kernel time; the "total prep+core" line is one graph with all three kernels.

Before the SOL pass (see "SOL pass" below for the current per-call numbers on GPU0):
GPU1 (no other compute process), SM clock 1500 MHz, precise state path, us, median of 50.
Production = chunk + plain scan + `gdn_gated_norm`; fused_norm = chunk + `gdn_scan` with the
fused norm (`GdnKernels(T, fused_norm=True)`):

| T | prep | chunk | scan plain | gated norm | core (production) | prep+core (production) | scan+norm fused | core fused_norm | prep+core fused_norm |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 8.77 | 14.62 | 13.06 | 7.52 | 28.67 | 33.60 | 14.98 | 26.18 | 31.58 |
| 256 | 11.49 | 20.29 | 17.70 | 9.28 | 39.42 | 48.13 | 20.19 | 36.45 | 44.61 |
| 512 | 15.84 | 29.98 | 26.14 | 12.42 | 63.87 | 75.33 | 31.62 | 58.94 | 71.26 |
| 1024 | 25.41 | 52.26 | 43.58 | 19.04 | 107.81 | 128.19 | 53.47 | 100.38 | 121.28 |

Cost of the unfused production path: core +2.5 / +3.0 / +4.9 / +7.4 us at T = 128 / 256 /
512 / 1024 (the extra launch, and o written and read once more through L2/HBM).

Amortised in a graph of 20 back-to-back calls (no launch latency): prep 4.9 / 21.2 us,
chunk 10.9 / 47.7 us, scan 10.9 / 48.8 us at T=128 / T=1024.

What limits it:
* `gdn_scan` is a serial recurrence over T/64 chunks with one math warpgroup per
  (head, 64 dv columns); per chunk ~3.8K cycles at T=1024, of which ~1.7K are tensor work;
  the rest is latency (bf16 conversions, Delta, scale) that one warpgroup cannot hide;
  ptxas holds it at 168 regs (240 cap) so a second state copy in registers is not possible.
* `gdn_chunk` per CTA (T=1024, ~18K cycles): G/beta load 2.4K, KK/QK 1.1K, A/P 1.9K,
  inverse 4.8K (f32 SIMT), U/W 1.6K, stores 1.8K; ~100 KB smem -> 2 CTAs/SM.
* At T=128 the three launches (dependent kernels, 2-CTA clusters) are a large share.

Rejected experiments: 512-thread chunk (slower), V load after the inverse (slower and an
allocator-aliasing hazard), prefetch of U into registers in the scan (spills), reordered O
gemm, a second accumulator for the P.Delta read-out to drop one wgmma wait (y rel_l2 went
to 8.1e-3 at T=1024 layer 60; not adopted), async final TMA stores in the chunk (no gain).

## SOL pass (per-call GPU time)

Code: runtime/experiments/gdn_sol.py (`--before` = the previous gdn.py); raw numbers:
reports/gdn_sol.json. GPU0 (SM 1500 MHz, slower than GPU1), no other compute process
(waited with a copy of the run’s GPU-idle helper that ignores the long-lived
`nvidia-cuda-mps-server` of another user on GPU0; the original script never returns while it
runs). Per-call time = runtime.benchutil.graph_time_us (20 back-to-back calls per graph),
before and after alternated in one process, median of 3 rounds. us per call:

| T | prep | chunk | scan_plain | gated_norm | sum |
|---|---|---|---|---|---|
| 128 | 5.06 -> 5.05 | 11.21 -> 10.65 | 9.40 -> 8.94 | 3.92 -> 3.49 | 29.59 -> 28.13 (-4.9%) |
| 256 | 7.74 -> 7.60 | 16.51 -> 15.78 | 13.62 -> 12.93 | 5.51 -> 4.47 | 43.38 -> 40.78 (-6.0%) |
| 512 | 12.02 -> 11.59 | 26.72 -> 25.05 | 21.95 -> 21.33 | 8.69 -> 6.84 | 69.38 -> 64.81 (-6.6%) |
| 1024 | 21.32 -> 20.45 | 48.34 -> 47.10 | 39.18 -> 38.76 | 15.43 -> 11.88 | 124.27 -> 118.19 (-4.9%) |

Changes (outputs: prep bitwise as before, y bitwise vs the HIR norm formula on the read-out
as before, state within the per-element contract):
* `gdn_gated_norm`: gate n2*silu(z) by ex2/rcp approx with a guarded exact fix-up (margin 16
  f32 ulps around a bf16 midpoint, or z < -16), checked over all 65280^2 finite bf16 (n2, z)
  pairs (0 mismatches; with margin 0 there are 1072); bf16 roundings as packed casts (the
  scalar cvt to bf16 ran on the XU pipe, 70% busy). Output bits unchanged.
* `gdn_prep`: the silu guard is one branch per thread (flag over the 4R values, midpoint test
  as one multiply-add + compare) instead of one per value; q/k/v stored straight from the
  packed bf16 registers. Output bits unchanged.
* `gdn_scan_plain`: no cluster launch (the plain path has no DSMEM traffic). Bits unchanged.
* `gdn_chunk`: inverse in 5 phases (diagonal 16x16 blocks, level 16->32 for blocks (1,0) and
  (3,2), level 32->64: T1 = A[32:64,0:32] X[0:32,0:32], X[32:64,0:32] = -X[32:64,32:64] T1)
  with 1x2 / 2x2 register tiles, instead of 7 phases with 1x2 tiles (fewer shared-memory
  wavefronts per FMA, all 256 threads busy). Different summation order: state rel_l2
  4.55e-6..6.42e-6 (was 4.53e-6..6.48e-6), worst abs/tol 0.020..0.032 (unchanged range).

Structure changes (scheduled HIR to regenerate): `gdn_scan_plain` lost `cluster_dims=(2,1,1)`
and the final cluster arrive/wait (roles, staging, barriers otherwise unchanged); `gdn_chunk`
inverse phases and their thread mappings (grid, threads, smem staging unchanged); `gdn_prep`
silu guard control flow and bf16 (not uint32) q/k/v parameters; `gdn_gated_norm` gate code
(grid unchanged). The fused `gdn_scan` (fused_norm=True) keeps its cluster.

Tried and rejected (numbers on GPU0):
* scan with 3 roles (state warpgroup; read-out warpgroup with Sh and bf16(Delta) handed over
  through smem, SS gemms and TMA store of o; producer): bitwise equal, 7.96 vs 8.68 us at
  T=128 and 12.36 vs 12.70 at T=256, but 21.48 vs 21.06 at T=512 and 39.56 vs 38.17 at
  T=1024 (the Sh handoff costs ~550 cycles per chunk). Not adopted (one structure for all T;
  it could serve T <= 256).
* scan with the dk dimension split over two state warpgroups (partial Y and Delta halves
  exchanged through smem, update with A from smem), 4 roles, 512 threads: 52.97 us at
  T=1024 (+35%).
* scan with Y and the update in two column halves to overlap Delta with the gemms: no change
  (~3.8K cycles per chunk). An RS wgmma issue holds the warp until the tensor core takes it,
  so SIMT work of the same warpgroup does not overlap its own gemms.
* scan hi/lo splits without cvt.rn.bf16x2 (truncated hi, or Veltkamp splitting on the FP32
  pipe): convert phase not faster; ~58% of its stall samples are no_instruction (instruction
  fetch of the long unrolled loop body), not the converter.
* scan skipping the gemms of chunk 0 (S = 0): the SIMT clear of the accumulators made ptxas
  serialise every wgmma (C7515).
* dropping one of the three Y terms: without Sl Wh worst abs/tol 2.0..2.5, without Sh Wl
  2.4..2.8 (fail); all three are needed for the state contract.
* chunk: early g/beta loads, asynchronous TMA stores, no X clear: 1..4% slower, reverted.
  (Hazard seen on the way: when the G_s/B_s smem writes moved before `T.wait_wgmma`, the
  allocator had placed them on top of Q_s, which the QK wgmma still read.)
* chunk with 384 or 512 threads: does not compile (no warp partition for a 64x64 wgmma over 12
  warps; fill layout of the (64, 68) tile over 512 threads).
* norm: 1..4 rows per warp, 4 or 8 warps, one fix-up branch per 8 values: all within 0.1 us.
* prep: other (rows, warps) configurations: equal or slower.

What limits them now:
* prep: ~31 instructions per element (conv 4 FFMA, two bf16 round trips, fast silu + guard,
  l2norm for q/k); issue active 72%, DRAM 26% at T=1024.
* chunk: per CTA ~14K cycles at T=128 (one CTA per SM, latency) and ~20K at T=1024 (2 CTAs per
  SM: 102 KB smem and 92 registers); inverse ~3K (shared-memory bandwidth), TMA load
  latency under load, conversions, store latency; 2.91 waves at T=1024.
* scan: serial over T/64 chunks, ~3.7K cycles per chunk on one math warpgroup; ~1.3K of it is
  the tensor work on the state chain (3 Y terms + 2 update terms), the rest SIMT (hi/lo
  split, Delta), instruction fetch and wgmma issue.
* norm: ~33 instructions per element, issue ~74%, DRAM ~48% at T=1024.

## Fusion alternatives (step-3 evidence)

Code: runtime/experiments/gdn_fusion.py; raw numbers: reports/gdn_fusion.json. Layer-0
inputs, GPU1 at 1500 MHz, GPU1 checked free of other compute processes before each T.
"graph" = the method of the timing table (one CUDA graph per piece, median of 50, includes
~3-4 us graph launch per graph); "sum" = sum of the per-kernel graph times; "chain" = all
kernels of the alternative in one graph; "b2b" = runtime.benchutil.graph_time_us (20
back-to-back calls in one graph, no launch latency). All times in us.

(A) prep: fused `gdn_prep` vs split = `small.conv_silu_kernel` + `small.l2norm_kernel` on q +
on k + `gb_window` (g, beta, conv window; minimal kernel in the experiment file). The split
outputs are bitwise equal to the fused ones (q, k, v, g, beta, window). The small kernels need
contiguous inputs; they got contiguous copies of `mixed` and of q/k (copies not timed).

| T | fused prep | conv_silu | l2norm q | l2norm k | g/beta/window | split sum | split chain | fused b2b | split chain b2b |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 8.61 | 10.59 | 6.34 | 6.50 | 5.44 | 28.87 | 17.89 | 5.06 | 13.73 |
| 256 | 11.55 | 16.16 | 7.78 | 7.78 | 5.57 | 37.29 | 25.92 | 7.59 | 21.68 |
| 512 | 15.78 | 26.88 | 10.34 | 10.46 | 5.66 | 53.34 | 42.43 | 12.02 | 38.61 |
| 1024 | 25.41 | 50.02 | 15.42 | 15.42 | 5.86 | 86.72 | 76.86 | 21.44 | 72.68 |

(B) gated RMSNorm: `gdn_scan` with the fused norm vs plain `gdn_scan` (z=None) + `gated_norm`
(minimal kernel in the experiment file: one warp per (t, head) row, HIR rounding, exact
silu). `gated_norm` output is bitwise equal to the HIR formula on the same read; the fused
norm differs from it in 4..30 of 0.8M..6.3M elements by 1 bf16 ulp (fast silu of z).

| T | scan fused | scan plain | gated_norm | plain+norm sum | plain+norm chain | fused b2b | chain b2b | core fused (chunk+scan) | core split chain |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 15.07 | 13.38 | 8.06 | 21.44 | 17.66 | 11.11 | 13.94 | 26.50 | 29.06 |
| 256 | 20.48 | 17.44 | 9.47 | 26.91 | 23.33 | 15.99 | 19.70 | 36.38 | 39.68 |
| 512 | 31.90 | 26.27 | 13.12 | 39.39 | 36.48 | 27.43 | 32.68 | 58.98 | 64.83 |
| 1024 | 53.38 | 43.46 | 19.74 | 63.20 | 60.51 | 48.93 | 56.22 | 100.70 | 107.87 |

(C) chunk + scan as one kernel vs two: not measured; no fused chunk+scan variant exists.

Reading: fused prep saves 9 us (T=128) to 51 us (T=1024) against the split chain; most of
the split cost is `small.conv_silu_kernel` (one thread per element, no vectorisation). The
fused norm costs the scan +1.7 us (T=128) to +9.9 us (T=1024) over the plain scan, but saves
2.6 us (T=128) to 7.1 us (T=1024) against plain scan + separate norm (one graph).
Decision after this study: plain scan + separate norm is the production path anyway (the
fused 2-CTA row-sum exchange cannot be stated in TileFoundry scheduled HIR). The norm kernel
moved into gdn.py as `gdn_gated_norm` (16-byte lanes, torch reduction order; slightly faster
than the experiment version: 19.0 vs 19.7 us at T=1024); the experiment now times that one.

## TileLang 0.1.15 facts (TILELANG_NOTES.md hypotheses checked, and new ones)

Confirmed on 0.1.15:
* Two sequential `T.ws` regions deadlock: a `__syncthreads()` is emitted between them
  (the producer never reaches it). Nesting the roles in the arms of one `if tx < 128 ...
  elif ... else` works (3 roles, 384 threads).
* A leading stage axis on the allocation (`T.alloc_shared((2, 64, 128))`, use `W_s[s]`)
  works for TMA loads and as a wgmma operand with a dynamic stage index.
* A TMA `T.copy` into a row slice of one swizzled buffer (`WQ_s[0:64, :]` of a (128,128)
  buffer) fails: "plain SMEM layout is not a bijection" -> one buffer per operand.
* Wrong barrier arrive counts hang or kill the context (missing arrive after `T.tma_copy`
  -> hang; one arrive too many on a cluster barrier -> CUDA_ERROR_LAUNCH_FAILED).
* Register budget in warp specialisation: with 384 threads ptxas caps at 168 regs and
  serialises every wgmma (C7512) until `T.inc_max_nreg(240)` / `T.dec_max_nreg(128)` are
  put in the roles; then no C75xx and 168/240 regs.
* Launch stream: kernels launch on torch's current stream; capture on a side stream
  (`torch.cuda.graph(g, stream=side)`) works and replays correctly.

Refuted / not reproduced on 0.1.15:
* "Layout inference uses the total thread count; threads=384 fails": not hit. 384 threads
  with all wgmma inside `T.ws(0)` (128 threads) compiles and the fragment layouts are those
  of one warpgroup.
* "T.Parallel binds to the kernel thread extent inside T.ws": not hit; T.Parallel inside
  `T.ws(0)` covered the 128 threads of that role.

New facts:
* `T.Pipelined` + `T.wgmma_gemm` (async): the automatic pipeline emits the consumer
  "empty" mbarrier arrive right after the wgmma issue, before `T.wait_wgmma`, so the
  producer may refill a stage that an in-flight wgmma still reads. Results were right in
  tests but the hazard is real; I replaced it by manual TMA double buffering.
* `T.tma_copy(..., barrier=b)` emits only `expect_tx` + load in one elected thread; the
  barrier (count 1) needs an explicit `T.mbarrier_arrive` by one thread, else the wait hangs.
* `T.copy_cluster(src, dst, dst_block=peer, remote_barrier=b)` with a contiguous 16B-multiple
  region lowers to remote `mbarrier.arrive.expect_tx` + `cp.async.bulk.shared::cluster`;
  the receiver's barrier count = number of incoming copies and the receiver must not add
  its own expect_tx. Without a barrier it lowers to SIMT `map_shared_rank` stores.
* A remote store + `fence.acq_rel.cluster` (via prelude) + remote arrive works but the fence
  waits for this thread's outstanding global stores; it made the fused norm 1.5x slower than
  the bulk-copy exchange.
* In one `T.Parallel` loop a fragment must be indexed identically (`Y[i, j]` and
  `Y[i, j + 64]` -> InternalError "Y: (i, j + 64) and (i, j)").
* 256 threads, `policy=FullRow` (M=128 over two warpgroups): the emitted
  `warpgroup_fence_operand(acc, n)` uses n = 2x the per-thread accumulator size; that
  per-head scan ran 5-10x slower than the 128-thread one; fixing the counts by a
  post-process did not restore speed (ptxas C7517 remained). Not used.
* ptxas C7517 (injected `wgmma.wait_group`) appears when the state is rescaled while other
  wgmmas are in flight; inside a warp-specialised (divergent) region this becomes C7518 and
  every wgmma is serialised. Rescaling S before issuing the chunk's wgmmas removes both.
* `T.Kernel(..., prelude=src)` injects device code; `T.call_extern("float32", "fn", x)` calls
  it (used for `ex2.approx.ftz` / `rcp.approx.ftz` in prep while the exact fallback keeps
  IEEE division and expf; prep is compiled without fast-math).
* Fast-math is off by default (`tl.enable_fast_math`); nvcc still contracts a*b+c to FMA,
  harmless for the conv because the f32 product of two bf16 values is exact.
* Fragment -> global `T.copy` of the chunk outputs (4-8 B per thread, strided) was the
  bottleneck of `gdn_chunk` (61 us at T=1024); staging through a smem tile annotated with
  `make_swizzled_layout` + TMA store brought it to 47 us.
* `T.clear` on a (64, 65) f32 shared tile fails (fill layout not bijective for 4160 elements);
  (64, 68) works and is bank-conflict free for the column walks used here.
* A Python conditional expression inside `T.unroll(...)` extent becomes a TIR
  `if_then_else` with a non-bool condition -> error; compute such counts outside.
* `from __future__ import annotations` in the kernel module makes `T.prim_func` fail to
  resolve closure variables in parameter annotations (NameError).
* ThreadSync prints "[ThreadSync] Hoisting sync out of an if ... Condition:
  (tx // 128 + d) // 2 + s < 2" (and similar) for `gdn_chunk`, in the block-distance loops of
  the inverse (d = block distance, s = work slot, e = s*256 + tx; the condition is
  e < (4 - d) * 128, uniform per warp). Each thread in the `if` writes only off-diagonal
  16x16 blocks (bi, bi-d) of X_s / Y_s; in the same phase all threads read only diagonal
  blocks and blocks of distance < d, written in earlier phases. Write set and read set are
  disjoint, and the hoisted barrier is reached by all 256 threads. It is not a real race;
  the per-element checks at all T confirm it.
* The shared-memory allocator reuses (aliases) buffers whose last *program-order* use is
  over; it does not know that an async `T.wgmma_gemm` still reads its operand until
  `T.wait_wgmma`. A buffer written after an un-waited wgmma can overwrite that wgmma's
  operand. Keep every wgmma operand live (use it after the wait) or wait first.
* An f32 shared tile read as a column of a wgmma-accumulator fragment (U^T in the scan) had
  8-way bank conflicts with the plain layout; `make_swizzled_layout` on the f32 tile (also
  the TMA destination) removed them.
