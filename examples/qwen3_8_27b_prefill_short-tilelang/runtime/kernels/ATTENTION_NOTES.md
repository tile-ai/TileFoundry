# Gated full-attention token mixer — tilelang kernels (Qwen3.8-27B prefill)

Files: `attention.py` (kernels + `AttentionKernels`), `test_attention.py` (correctness,
cache slices, CUDA graph, timing), this file. Build: tilelang 0.1.15, torch 2.14.1+cu130,
CUDA 13.2, H200 GPU0, SM clock 1500 MHz (read with nvidia-smi before and after each timing).

Run: `CUDA_VISIBLE_DEVICES=0 python runtime/kernels/test_attention.py` (the per-kernel
test scripts this file and GDN_NOTES.md name are in the round archive, not here)
(`--T 128 1024`, `--layers 3 63`, `--no-timing`, `--json out.json`).

## API

```python
from runtime.kernels.attention import AttentionKernels
ak = AttentionKernels(T)                       # T in (128, 256, 512, 1024); compiles 2 kernels
ak.prep(qkv, gamma_q, gamma_k, cos, sin, q, k, v)
ak.core(q, k, v, qkv, y)                       # gate read from qkv
```

| tensor | shape | notes |
|---|---|---|
| qkv | (T, 14336) | per row `[24 x (query 256, gate 256)] [k 4 x 256] [v 4 x 256]` |
| gamma_q, gamma_k | (256,) | |
| cos, sin | (T, 64) | |
| q | (T, 24, 256) | written by prep, read by core |
| k, v | (T, 4, 256) | written by prep. **v is a copy** of `qkv[:, 13312:]` (so it can land in the cache) |
| y | (T, 6144) | written by core, `y[t, h*256 + d]` |

All bf16, CUDA, row-major contiguous; the wrappers check shape, dtype and
`is_contiguous()` (`AttentionKernels(T, check=False)` skips the checks). K/V outputs
may be slices of a `(16, Tmax, 4, 256)` cache: `k_cache[i]` and `k_cache[i, :T]` are
contiguous `(T, 4, 256)` views, which is what the kernels need (tested with slot 5 of a
16-slot cache; other slots stay untouched). Arbitrary strides are not supported.
Both kernels launch on torch's current stream (TVM-FFI reads it via DLPack at call
time) and are CUDA-graph capturable (tested: capture of prep+core, replay with new
inputs copied into the static buffers is bitwise equal to eager runs).

## Kernel A: `attn_prep`

Grid T CTAs x 1024 threads; warp w of CTA t handles slot w of token t:
0..23 query head, 24..27 key head, 28..31 value copy. Lane l owns elements
`4l..4l+3` and `128+4l..128+4l+3` (two 8-byte loads).

Bitwise equality with the HIR (q, k: 0 mismatching elements at every T, layers 3 and 63)
needs the f32 sum of squares in **exactly torch's reduction order** for
`x.float().pow(2).mean(-1)` on a contiguous (rows, 256) tensor, found by emulation
(24576/24576 rows matched, the other 5 candidate orders mismatched ~35 % of rows):
`a_i = x[4l+i]^2 + x[128+4l+i]^2`, lane sum `((a0+a1)+a2)+a3`, then `shfl_down`
16, 8, 4, 2, 1, times 1/256 (exact). Every f32 op is `T.ieee_mul / T.ieee_add`
(`__fmul_rn / __fadd_rn`, so no FMA contraction); `T.rsqrt` -> `rsqrtf` gives the same bits
as `torch.rsqrt`. Rope in bf16 exactly as the HIR: `bf16(bf16(x*cos) + bf16(rot(x)*sin))`,
partner element by `shfl_xor(., 8)`. Compiled without fast math. 32 registers.

## Kernel B: `attn_core` (hand warp-specialized, `make_core`)

CTA = (1 or 2 query heads of one KV group) x 64 query tokens x (256 / DSPLIT output
dims). Warpgroup 0 = producer, warpgroup 1 + w = consumer for head `ha + w`.
Grid `(24/HPC*DSPLIT, T/64)`, heads on the fast axis, token blocks from last to
first (longest causal range first = LPT order for the block scheduler).

- Producer: warp 0 TMA Q (+ K ring, ST=2 stages of 64x256), warp 1 TMA V ring
  (ST=2), mbarrier full/empty pairs (full: 32 arrivals, empty: 128 per consumer WG).
- Consumer (FA3-style intra-warpgroup overlap with explicit `T.wgmma_gemm` +
  `T.wait_wgmma`): issue `S_{j+1} = Q K_{j+1}^T`, rescale O, issue `O += P_j V_j`,
  wait S, softmax(S) while PV runs, wait PV, `P = bf16(exp2(S))`.
  Scores f32 scaled by `log2(e)/16`, running max/sum f32, P in bf16 for the PV wgmma,
  f32 accumulation, `O/l` landed in bf16 — the HIR `attend` rounding.
- Gate: `y = bf16(bf16(O/l) * bf16(sigmoid(gate)))`, TMA store.
  - "early": warp 1 TMA-loads the gate tile into its own buffer first; producer warps
    compute bf16(sigmoid) in place while the consumer runs (all 4 warps when
    T <= 64*ST so their TMA loops never block, else warps 2-3); the epilogue only
    multiplies and stages y in the gate buffer.
  - "late": warp 0 loads the gate of head `ha+w` into K-ring slot `nkb+w` (the stage
    freed by K tile `nkb-2+w`); the consumer applies the sigmoid in its epilogue
    (in place in shared memory) and stages y in its Q tile.
  - DSPLIT=2: the two CTAs of a (head, token block) both compute S and the softmax;
    each does P V, the gate and the store for 128 of the 256 output dims.
- Config per T (`CORE_CONFIG`, chosen by measurement):

| T | heads/CTA | gate | DSPLIT | CTAs | threads | dyn. smem | regs (ptxas) |
|---|---|---|---|---|---|---|---|
| 128 | 1 | early, 4 warps | 2 | 96 | 256 | 144 KB | 144 |
| 256 | 1 | early, warps 2-3 | 1 | 96 | 256 | 192 KB | 239 |
| 512 | 1 | late | 1 | 192 | 256 | 160 KB | 205 |
| 1024 | 2 | late | 1 | 192 | 384 | 192 KB | 168 launch, setmaxnreg 40/232 |

  No spills anywhere; 40 (80 for 2 heads) HGMMA with 4 (8) DEPBAR, i.e. no wgmma
  serialization. All compiled with `tl.enable_fast_math` (ex2.approx, rcp.approx);
  the early-gate kernels also with `tl.disable_thread_storage_sync` and explicit
  `fence_proxy_async + bar.sync(8+w, 128)`.

## Correctness (test_attention.py, HIR = TileFoundry evaluation of model.py)

Inputs are real: `x = rms_norm(layer{i}_hidden)`, `qkv = [x w_q^T | x w_k^T | x w_v^T]`
(bf16 torch matmuls), cos/sin from `reference/T{T}`. q/k refs = `lay.partial_rope(_kv)`
of the evaluator's rms_norm; y ref = `lay.attend(q,k,v) * sigmoid(gate)` (bf16 ops);
the split was checked against `lay.gated_attention` (its k, v equal ours bitwise).

| T | layer | q/k/v mismatches | y rel_l2 | y cosine | y max-abs (max |y_ref|) | o_proj(y) vs HIR out rel_l2 |
|---|---|---|---|---|---|---|
| 128 | 3 | 0/0/0 | 4.44e-4 | 0.99999988 | 0.0039 (3.95) | 1.7e-4 |
| 128 | 63 | 0/0/0 | 4.94e-4 | 0.99999994 | 0.0625 (38.5) | 8.7e-4 |
| 256 | 3 | 0/0/0 | 5.77e-4 | 0.99999988 | 0.0039 (3.94) | 7.2e-4 |
| 256 | 63 | 0/0/0 | 6.70e-4 | 0.99999970 | 0.125 (83.5) | 1.2e-3 |
| 512 | 3 | 0/0/0 | 7.29e-4 | 0.99999976 | 0.0078 (3.94) | 6.1e-4 |
| 512 | 63 | 0/0/0 | 7.72e-4 | 0.99999976 | 0.125 (74) | 1.3e-3 |
| 1024 | 3 | 0/0/0 | 8.25e-4 | 0.99999964 | 0.0078 (3.94) | 7.1e-4 |
| 1024 | 63 | 0/0/0 | 8.12e-4 | 0.99999964 | 0.125 (82.5) | 1.3e-3 |

Bounds: rel_l2 <= 0.009, cosine >= 0.99996 — all pass. Determinism: 100 repeated
prep+core runs per T (300 in a separate stress run) reproduce the first result bitwise.
The test was run 4 times in a row: all pass. The max-abs values are one bf16
ulp of the largest outputs (P lands in bf16 relative to the running max instead of the
global max, and the sigmoid uses ex2/rcp approx).

## Timing (GPU0, 1500 MHz, median of 50 after 10 warm-up, CUDA events)

Each timed launch is preceded by a `torch.cuda._sleep` spin kernel so the events time the
GPU only (the TVM-FFI call costs ~7 us of host time). Floor of this method: an empty
kernel measures 4.3 us, two empty kernels 5.7 us, a graph of two empty kernels 5.2 us.

| T | prep | core | core TFLOP/s | prep+core | graph(prep+core) |
|---|---|---|---|---|---|
| 128 | 6.6 [6.1..6.8] | 10.2 [9.7..10.4] | 20 | 14.0 | 13.2 |
| 256 | 7.7 [7.5..8.1] | 16.0 [15.8..16.4] | 51 | 20.6 | 19.9 |
| 512 | 10.3 [10.0..11.1] | 26.2 [25.9..26.7] | 123 | 33.6 | 33.0 |
| 1024 | 15.5 [15.2..16.2] | 45.3 [43.9..46.1] | 284 | 60.2 | 59.9 |

(us, median [min..max] of 50; FLOP = causal pairs only, 24 x 2 GEMMs x 2 x 256 x
T(T+1)/2 = 12.9 GFLOP at T=1024; bf16 dense peak at 1500 MHz is ~811 TFLOP/s.)
SM clock read 1500 MHz before and after every row. GPU0 is shared: an SGLang process of
another agent was resident during this final run (GPU utilization samples 0..48 %);
the numbers match earlier runs taken with GPU0 idle within ~0.5 us. During part of the
development a concurrent job doubled kernel times; those runs were discarded.
Subtracting the floor, real work at T=128 is about 2.3 us (prep) + 6 us (core).

### What limits it

- prep: memory. T=1024 moves 33.5 MB (q, k, v read + written; the gate half of q_proj is
  not read) in ~11 us kernel time = ~3 TB/s. T=128 is latency (2.2 us above the floor).
- core T=128: latency of one CTA (96 CTAs, ~6 us kernel). Before the output-dim split
  (48 CTAs) the globaltimer stamps were: TMA Q+K0 arrive ~1.4 us, first QK+softmax ~1 us,
  one more block ~1.4 us, tail PV 0.6 us, gate multiply 1.1 us, TMA store 0.8 us; the
  split halves the PV, gate and store part per CTA and uses 96 SMs.
- core T=1024: the 16-block CTAs (last 64 query tokens, 2 heads) are the critical path:
  16 x ~1.7 us per KV block + ~9.5 us fixed per CTA (first-wave Q/K load burst
  ~2.5-3.3 us, first QK+softmax ~1.5 us, tail ~0.7 us, gate ~4 us, store ~1 us). Per-block cost measured with a fixed-iteration variant (132
  CTAs, no reloads): 2493 cycles; pure wgmma (no softmax) 2124; without the O rescale
  2280; the two-warpgroup wgmma alone 2069 (= 803 TFLOP/s). So the in-loop softmax +
  rescale cost ~20 %, and the fixed per-CTA prologue/epilogue ~25 %. Per-CTA timeline
  (final kernel, 2 heads): CTA duration = ~9.5 us + 1.7 us per KV block; of the fixed
  part the epilogue sigmoid over 2 x 64 x 256 gate elements takes ~4 us (SFU bound
  2.7 us: 2 MUFU per element). The kernel span (40.4 us) is set by SMs that run two
  CTAs (e.g. 9 + 2 blocks = 26.7 + 13.1 us), i.e. by the fixed cost paid twice; the
  single 16-block CTAs end at 36.7 us. Memory traffic of the core at T=1024 is 42 MB
  (q, gate, y 12.6 MB each, K/V 4 MB) = ~14 us at 3 TB/s. A persistent kernel that
  overlaps one item's epilogue with the next item's loads would save an estimated
  3-4 us (bounded by the 16-block CTAs); not built.

## What was tried (core kernel; us, event-timed, same method)

| variant | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| auto: `T.gemm` + `T.Pipelined` auto-WS, BM=128 BN=64, IEEE div/expf epilogue | 31.7 | 36.4 | 45.3 | 112.8 |
| + fast sigmoid (`T.fast_rcp`), gate/out staged in smem | 16.0 (BM64) | | | 78.4 |
| + heads on the fast grid axis (real LPT order) | 15.9 (BM64) | 18.9 (BM64) | 30.6 (BM64) | 55.8 (BN128) |
| auto, BM=64 with 2 consumer WGs splitting columns, P via smem | 24.8 | 29.1 | | |
| hand WS, 1 head/CTA, first version | 16.1 | 22.1 | 35.1 | 85.4 |
| + P converted after the PV wait (removes C7513 wgmma serialization) | 17.2 | | | 72.3 |
| + K / V / gate on separate producer warps (K no longer waits behind V) | 17.0 | | | 61.8 |
| + fast-math sigmoid without `asm volatile` rcp, epilogue via register fragment | 15.2 | 17.1 | 28.4 | 55.7 |
| hand WS, 2 heads/CTA (two consumer WGs share each K/V tile) | 20.6 | 23.2 | 31.1 | 49.7 |
| 2 heads, one shared consumer code path (generic Q loads) | | | | 50.7 |
| 2 heads, gate TMA into the freed K ring slots, fast math | 19.6 | 23.5 | 30.6 | 48.4 |
| gate slots + consumer sigmoid in smem (final "late" mode) | 13.6 | 16.5 | **26.3** | **45.2** |
| own gate buffer, sigmoid by producer warps during the loop (4 warps at T=128, warps 2-3 at T>=256) | 12.6 | 16.4 | 27.1 | 45.4 (head A only) |
| FA3 ping-pong named barriers between the two consumer WGs | | | | 45.6 |
| max kept unless it grows by 2^8 + warp-vote skip of the O rescale | 13.8 | | | 44.7, rel_l2 doubled: rejected |
| "late" mode with `tl.disable_thread_storage_sync` | 13.8 | | | 47.8 |
| early gate + output dims split over 2 CTAs (DSPLIT=2, final at T=128) | **10.0** | 17.0 | 29.9 | |
| same, DSPLIT=4 | 11.8 | 19.5 | 43.9 | |
| own gate buffer + warps 2-3 sigmoid, DSPLIT=1 (final at T=256) | 12.4 | **16.0** | 26.8 | |
| late gate, sigmoid with one MUFU (ex2 + 3 Newton steps for 1/x) | | | 28.3 | 55.5 |

Parallelism notes: at T=128 one head per CTA gives 48 CTAs (two heads: 24 -> slower),
splitting the output dims over 2 CTAs gives 96 CTAs and the best time (QK^T and softmax
are recomputed, PV/gate/store halve);
packing the 6 heads of a GQA group into one CTA does not reduce the causal MMA work and
needs a 6-head Q tile that TMA cannot land as one swizzled 2-D tile, so it was not built;
2 heads/CTA halves L2->SM K/V traffic and wins only at T=1024. Split-KV was not tried
(needs a combine pass); the largest CTAs (16 KV blocks) remain the T=1024 critical path.

## TileLang facts on 0.1.15 (vs ROOT/TILELANG_NOTES.md)

Hit / confirmed:
- Two sequential `T.ws` regions deadlock: a `__syncthreads()` is emitted between them and
  the producer (blocked on an empty barrier) never reaches it. `if tx < 128: with T.ws(0)
  ... else: with T.ws(1) ...` works.
- WGMMA serialization (ptxas C7513, every HGMMA followed by `WARPGROUP.DEPBAR`): new
  trigger — copying a second bf16 register buffer into the PV A-operand while the PV
  wgmma is in flight. Converting S -> P after `wait_wgmma(0)` (FA3 order) fixes it
  (40 HGMMA / 4 DEPBAR).
- R2UR before WGMMA is present (54..94 R2UR per kernel); not removed (not the bottleneck).
- The compiled kernel object reports nothing: `n_regs None`, `n_spills None`,
  `resource_usage {}`; use ptxas -v on `get_kernel_source()`.
- Swizzled shared buffers at an offset: TMA into a row slice `Qs[0:64, :]` of a swizzled
  (128, 256) buffer fails ("plain SMEM layout is not a bijection"); `T.view` of a
  swizzled 3-D buffer as 2-D fails ("Get different layout"); a TMA box (64, 2, 256)
  into a 2-D (128, 256) tile is refused (shape mismatch). A leading stage axis
  (`Qs[w, :, :]`, `Ks[s, :, :]`, dynamic `s`) works for TMA, wgmma operands and TMA store.
- `T.dec_max_nreg / T.inc_max_nreg` emit setmaxnreg; the auto-WS emits
  `warpgroup_reg_dealloc<24>` / `reg_alloc<240>`.
- `T.Parallel` inside a `T.ws(i)` scope binds to that warpgroup's 128 threads
  (used in producer and consumers, results correct).
- High-level `T.gemm` + `T.Pipelined` auto-WS is correct but slower than the hand WS:
  55.8 vs 45.3 us at T=1024, 15.9 vs 10.2 us at T=128.

Refuted / not reproduced:
- "threads=384 hangs / layout inference uses the total thread count": a 384-thread kernel
  with `T.wgmma_gemm` in `T.ws(1)` and `T.ws(2)` (separate fragments), and also in a joint
  `T.ws(1, 2)` scope (M=128, FullRow), compiled and ran correctly.
- "Graph capture fails ~1 in 3 (kernel not on the calling stream)": with the default
  TVM-FFI backend every capture (>= 8 per test run) succeeded and replays are bitwise equal
  to eager runs.

New:
- `T.tma_copy` (load) only does expect_tx + copy: the barrier needs an explicit
  `T.mbarrier_arrive` (count = threads of the scope that arrive). Missing it hangs.
- `T.tma_copy` elects its issuing lane with `tl_shuffle_elect<scope threads>`, i.e. warp
  index % 4 == 0 for a warpgroup scope: TMA issued from warp 1 inside a warpgroup branch
  needs `leader_scope_threads=32` (read from intrin.h; used, failure case not run).
- `T.fast_rcp` is `asm volatile("rcp.approx...")`: it pins instruction order; an epilogue
  with one per element cost ~6.5 us per CTA. `1.0 / x` with `tl.enable_fast_math`
  (off by default in 0.1.15) schedules freely.
- Automatic shared-memory sync insertion puts a `bar.sync` inside every iteration of an
  in-place read-modify-write loop over a shared buffer, and before every element store
  into a buffer that a wgmma read earlier (64 barriers in one epilogue). Computing into a
  register fragment and doing one `T.copy` avoids the second case;
  `tl.disable_thread_storage_sync` + explicit `T.fence_proxy_async()` +
  `T.sync_threads(id, 128)` avoids the first.
- `T.macro` must be defined outside the `prim_func` (NameError otherwise), and a macro
  cannot assign through a `BufferRegion` argument (`'BufferRegion' object does not support
  item assignment`): pass buffer + indices.
- `prelude=` text must start with a newline, otherwise it lands inside the generated MSVC
  `#if` block ("#endif without #if").
- With `from __future__ import annotations` in the module, `T.Tensor((T_, ...))` closure
  annotations fail to resolve (NameError: T_) — do not use it in kernel modules.
- TileLang's on-disk kernel cache makes recompiling an unchanged kernel ~0.3 s.
