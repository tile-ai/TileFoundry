# GEMM_WS_NOTES — hand warp-specialized GEMMs (runtime/kernels/gemm_ws.py)

Build: tilelang 0.1.15, H200 GPU1, SM clock locked 1500 MHz. Time = CUDA-graph replay of 20 calls
(runtime/benchutil.py). All numbers in this file come from reports/gemm_ws*.json(l).

## Design

- 384 threads, 1 CTA/SM. Warpgroup 0 = producer: warp 0 issues every TMA load
  (`T.tma_copy(..., leader_scope_threads=32)` + `T.mbarrier_arrive`, full barrier count 32);
  warp 1 = optional IO warp. Warpgroups 1-2 = consumers in one joint `T.ws(1, 2)` scope:
  one BM=128 tile, FullRow (warpgroup g owns rows 64g..64g+63), `T.wgmma_gemm` + `T.wait_wgmma(0)`
  per k step, then lane 0 of each consumer warp arrives on the empty barrier (count 8).
- Ring: `stages` slots `[S, rows, 64]` with a leading stage axis + `make_swizzled_layout`.
- Same K order as gemm_kernel (one f32 accumulator, k16 steps ascending) and same epilogue
  operations, so outputs are bitwise equal to gemm_kernel with the same split.
- Options (WsConfig): `grid` (persistent tile loop, M-fastest order), `pk` (packed casts),
  `amerge` (staging tile aliases the A ring), `io` (IO warp prefetches R and issues stores),
  `csk` (split-K 2 inside a 2-CTA cluster, partial sent through DSMEM), `direct` (no staging).
  Experimental, measured, not used: `CL`/`mc` (TMA multicast), `wait1`, `wbox`, `pdl`, `debug`.
- `WS_CONFIGS[T][linear]` = (WsConfig, epilogue) or None (keep gemm_kernel). Calls: store
  `k(A, W, C)`, swiglu2 `k(A, Wg, Wu, O)`, residual `k(A, W, R, C)`. The csk residual kernel replaces
  today's "splitk" GEMM + splitk_reduce_kernel at T=128 (no P buffer needed).

## Results (reports/gemm_ws.json; final run = 5 interleaved rounds, paired ratio new/today)

Rule: use gemm_ws where the paired ratio <= 0.990. "today" = same-run median (2-6 % slower than the
numbers in the task brief: card state on this day). Bitwise: 9 cases (3 seeds x input scale 0.1/1/30,
scale 30 overflows exp in SwiGLU) + 200 repeated launches, 0 mismatches, for every replaced entry.

| linear | T | today us | new us | ratio | config |
|---|---|---|---|---|---|
| gate_up | 128 | 94.68 | (94.08) | 0.996 | keep |
| gate_up | 256 | 152.95 | 146.45 | 0.958 | BN144 s4 pk amerge |
| gate_up | 512 | 268.66 | 255.19 | 0.950 | BN176 s3 persistent132 pk |
| gate_up | 1024 | 520.56 | 497.97 | 0.956 | BN176 s3 persistent132 pk |
| down | 128 | 54.95 | 53.88 | 0.981 | BN80 s6 csk io pk (split 2 in cluster) |
| down | 256 | 78.25 | (78.34) | 0.994 | keep |
| down | 512 | 134.58 | (134.15) | 0.992 | keep |
| down | 1024 | 265.49 | 262.59 | 0.990 | BN160 s4 persistent132 pk |
| o_proj | 128 | 25.74 | 24.14 | 0.942 | BN80 s6 csk io pk |
| o_proj | 256 | 35.48 | (35.71) | 1.011 | keep |
| o_proj | 512 | 50.42 | 49.09 | 0.973 | BN160 s4 io pk |
| o_proj | 1024 | 99.27 | 96.66 | 0.974 | BN160 s4 persistent132 pk |
| gdn_in | 128 | 46.69 | (47.33) | 1.006 | keep |
| gdn_in | 256 | 67.08 | (66.69) | 0.993 | keep |
| gdn_in | 512 | 125.77 | 122.35 | 0.981 | BN256 s3 persistent132 |
| gdn_in | 1024 | 246.08 | 239.57 | 0.974 | BN256 s3 persistent132 |
| att_in | 128 | 41.15 | (41.34) | 1.006 | keep |
| att_in | 256 | 57.60 | (58.03) | 0.992 | keep |
| att_in | 512 | 110.83 | (110.89) | 0.992 | keep |
| att_in | 1024 | 217.13 | 211.60 | 0.975 | BN224 s3 persistent132 |

Estimated saving per forward (call counts 64 gate_up / 64 down / 64 o_proj / 48 gdn_in / 16 att_in):
T=1024 ~2.2 ms, T=512 ~1.1 ms, T=256 ~0.4 ms, T=128 ~0.17 ms.

## Directions

Worked:
- Persistent cooperative kernel (a): T=1024 gdn_in 0.974, att_in 0.975, o_proj 0.974, down 0.990;
  T=512 gdn_in 0.981. Gain = epilogue of tile i overlapped by the producer prefetching tile i+1 and
  the deferred TMA-store wait. No gain for single-round grids (T<=256 store GEMMs).
- Packed epilogue casts (pk): the SwiGLU epilogue issued 2 scalar F2F + 2 MUFU per element (both
  quarter rate: 256 x 335 / 16 = 5360 clk = 3.6 us per tile, as measured). `T.copy` fragment ->
  fragment emits F2FP.BF16.F32.PACK_AB; gate_up T=1024 508.1 -> 496.4 us, bitwise.
- csk (cluster split-K with DSMEM reduction): T=128 o_proj 0.942, down 0.981, bitwise vs split-K +
  reduce. At T=256 (BN160) the 80 KB receive tile leaves 3 or 2 ring stages: 1.19 / 1.48 (worse).
- amerge (staging tile aliased with the A ring): T=256 gdn_in s3 1.045 -> s4 0.993; needed for
  non-persistent hand-WS kernels to match the auto pipeline's stage count.

Did not work (stopped):
- Pingpong (two consumer warpgroups on alternate tiles): not built. 64-row tiles per warpgroup
  raise L2->SM bytes per FLOP by 1.67-1.74x, but gate_up's load-only variant (no MMA) already takes
  503 us of the 499 us kernel; 128-row tiles per warpgroup need 2x accumulator registers (BN176:
  352 regs). ncu, gate_up T=1024: tensor pipe 95.0 % active with the epilogue, 98.4 % without
  (475.7 us): the mainloop is at the tensor-core limit, the SwiGLU epilogue (2 MUFU per element,
  required for bitwise equality) costs the remaining ~4 %.
- TMA multicast of A, 2-CTA cluster along N (b): T=128 gdn_in 47.2 vs 46.5 us, down 57.0 vs 55.1;
  CL=4: 75.3 / 93.2 us (clusters of 4 do not all fit in one wave). At T=128 the limit is the W
  stream from HBM, not A: gdn_in full 46.5 us, without A loads 45.4, W loads only 43.1, without W
  loads 34.4. A GEMV on the same bytes takes 42.1 (gdn_in), 91.8 (gate_up), 52.2 (down),
  21.0 us (o_proj): today's T=128 kernels are 3-11 % above that (o_proj 23 %, now 15 %).
- TMA multicast of W, cluster along M: gate_up T=1024 505.3 vs 496.1 us.
- IO warp for persistent kernels: gdn_in T=1024 1.007-1.011 vs 0.974 without (also with one arrive
  per warp). Kept only for non-persistent residual tiles (o_proj T=512 0.973 vs 0.993).
- W tile as several TMA boxes (wbox 128): gate_up 1024 0.983 vs 0.949, att_in 1024 0.980 vs 0.973.
- wait_wgmma(1) with delayed slot release: gdn_in 1024 249.4 vs 239.2 us.
- PDL with W prefetch before griddepcontrol.wait: 16-layer [gate_up -> down] chain, us/layer
  median today / new / new+PDL: T=128 155.0 / 153.3 / 152.2, T=256 235.1 / 229.8 / 232.9; minima
  149.1 / 150.4 / 151.8 and 227.4 / 227.8 / 227.7: within the +-2 % noise. Bitwise. Not used.
- HB=136 (gate_up 128 CTAs at T=128): 17 x m64n8k16 per k16, 281 / 533 us (3-3.7x slower).
- BK=128 for gdn_in T=128: 46.95-48.1 vs 46.1 us.

## TileLang 0.1.15 facts

Confirmed (TILELANG_NOTES / ATTENTION_NOTES / GDN_NOTES):
- Hand WS with `if tx < 128: T.ws(0) ... else: T.ws(1, 2)` and FullRow `T.wgmma_gemm` at 384
  threads works (no hang, correct layouts).
- `T.tma_copy` load needs an explicit `T.mbarrier_arrive`; `leader_scope_threads=32` gives
  `elect_one_sync()` in the issuing warp; this also holds for TMA stores (IO warp).
- Leading stage axis on ring buffers works for TMA and wgmma operands.
- `T.copy_cluster(src, dst, dst_block=0, remote_barrier=b)` lowers to one thread issuing
  `tma_store_cluster` (bulk DSMEM copy), receiver barrier count 1.
- R2UR is not a problem here: 2 R2UR per HGMMA, 18 in the kernel, no spills.
New:
- `T.tma_copy(..., cluster_mask=0b11)` multicast works in a hand pipeline (bitwise, no hang) when
  the empty barriers are `T.alloc_cluster_barrier([8 * CL])` and lane r of every consumer warp
  arrives on CTA r (`T.mbarrier_arrive(empty[s], r)`): this is what the automatic pipeline lacks.
- MergeSharedMemoryAllocations does not alias a staging tile with the ring across the
  producer/consumer if-arms; `tl.enable_aggressive_shared_memory_merge` does. The aliasing is
  only valid for one tile per CTA, and it needs a consumer barrier before the staging write: no
  sync is inserted for the async wgmma that still reads the ring (found as a chain mismatch).
- Scalar f32 -> bf16 casts in `T.Parallel` emit F2F (quarter rate); `T.copy` fragment ->
  fragment emits packed F2FP. Same rounding.
- `T.fast_rcp` is `asm volatile rcp.approx.ftz.f32`; a prelude function with the same
  instruction without `volatile` (`T.call_extern`) is bitwise equal.
- wgmma N is widened only for N % 16 == 0 (`select_wgmma_inst_n`); N=136 -> n8 atoms, also for
  `T.wgmma_gemm`.
- Load-only rate (TMA, persistent, T=1024 shapes) depends on the W box height: 256 rows 13.9 TB/s,
  128 rows 11.3, 144-240 rows 8.0-9.0 TB/s; two boxes of <=128 rows do not restore it.
- 256 threads arriving on one mbarrier per tile is slow; one lane per warp after `T.sync_warp()`.
- `T.pdl_trigger()` / `T.pdl_sync()` emit `cudaTriggerProgrammaticLaunchCompletion()` /
  `cudaGridDependencySynchronize()`; TMA loads before `pdl_sync` compile and run.
- tilelang re-reads macro source from the .py file at compile time: editing the kernel file
  while another process compiles from it breaks that compile (sweeps run from a snapshot copy).
- ncu works here (counters accessible). Under tensor load the SM clock reads 1.42-1.46 GHz
  although locked at 1500 MHz, so the tensor peak during GEMMs is ~770-790 TFLOP/s, not 811.

## Files

- runtime/kernels/gemm_ws.py: kernels, `WS_CONFIGS`.
- runtime/experiments/gemm_ws_try.py (one config vs today), gemm_ws_sweep.py (interleaved paired
  timing, snapshot of the kernel file), gemm_ws_bitwise.py, gemm_ws_repeat.py, gemm_ws_chain.py
  (PDL chain), gemm_ws_collect.py.
- reports/gemm_ws.json (summary), gemm_ws_final.jsonl, gemm_ws_sweep1/2.jsonl,
  gemm_ws_bitwise.json, gemm_ws_repeat.json, gemm_ws_chain.jsonl.
