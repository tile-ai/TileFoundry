# Qwen3.8-27B prefill on TileFoundry + tilelang: results

> **This file is the agent's own report, kept whole.** `README.md` is the front
> door; everything below is what the run wrote about its own numbers. Two kinds
> of path in it do not resolve here: `evidence/` and most of `results/` are the
> run's working directories, which are kept in the round archive and not
> shipped. The subset every number in section 1 rests on is in `reports/`, and
> `tir/` keeps one representative per distinct lowering (`tir/INDEX.md`). The
> runtime twin is `runtime_model.py` here; the run called it `runtime.py`.

Text prefill only: batch 1, no history, prompt 1024 / 4096 / 8192 tokens, stop
after the first token. bf16 weights, no quantization, no vision, no decode.

## 1. The table

Measured on the pinned card (`CUDA_VISIBLE_DEVICES=2`, H200
GPU-777ba55d, the card of SGLANG_BASELINE.md), `tools/bench.py`,
2026-10-13 07:45, no other process on the card during the run
(`reports/bench-final-v9-gpu2.json`). This is the build whose kernels follow
the finalized TIR of every function, after this round's alignment (section 7;
STEP4 rows 27-43). Builds before it on the same card, this round:
80.26 / 308.94 / 630.82 ms (before row 43, `reports/bench-final-v8-gpu2.json`),
80.50 / 309.04 / 631.22 ms (before row 41, `reports/bench-final-v7-gpu2.json`),
80.53 / 308.54 / 629.95 ms (before row 39, `reports/bench-final-v6-gpu2.json`),
80.46 / 309.48 / 630.39 ms (before row 36, `reports/bench-final-v5-gpu2.json`); the
previous round: 80.48 / 310.05 / 632.16 ms (`reports/bench-final-v3-gpu2.json`);
before any alignment: 80.46 / 307.62 / 629.61 ms (`reports/bench-final-gpu2.json`).

| prompt | this work, median (range of 7) | SGLang 0.5.21, column B | vs B | compute floor (FLOP / 989 TFLOP/s) | this work TFLOP/s | % of 989 | SM clock, median (7 readings) | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | **80.29 ms** (80.02 - 82.61) | 89.06 ms | -9.8 % | 51.5 ms | 634.6 | 64.2 % | 1500 MHz (1470-1500) | 12 754 |
| 4096 | **309.61 ms** (306.90 - 315.89) | 317.11 ms | -2.4 % | 208.6 ms | 666.2 | 67.4 % | 1500 MHz (1455-1500) | 13 229 |
| 8192 | **631.19 ms** (629.11 - 670.55) | 638.25 ms | -1.1 % | 423.8 ms | 664.1 | 67.1 % | 1440 MHz (1365-1500) | 12 979 |

FLOP, floor and TFLOP/s in this table are `flops.py`'s, as the baseline's
are, so the rows compare. `flops.py` overcounts by 1.7 % (section 4); with the
corrected count the three rows are 50.08 / 202.78 / 412.15 TFLOP, floors
50.6 / 205.0 / 416.7 ms, and 623.7 / 654.9 / 653.0 TFLOP/s.

The same rows against the other denominators (section 5) and the other
column (section 3):

| prompt | % of 989 x clk / 1980 (the baseline's) | % of 132 x 4096 x clk (measured peak at the clock) | corrected FLOP at the clock: floor | vs SGLang column A |
|---:|---:|---:|---:|---:|
| 1024 | 84.7 % | 78.2 % (corrected FLOP: 76.9 %) | 61.8 ms | -7.6 % (86.94 ms) |
| 4096 | 88.9 % | 82.1 % (80.8 %) | 250.0 ms | -3.2 % (319.94 ms) |
| 8192 | 92.3 % | 85.3 % (83.9 %) | 529.4 ms | -9.4 % (696.38 ms) |

The 8192 row's margin to column B is 7.1 ms (1.1 %), and its range of 7
reaches 671 ms: the first timed replay is the slow one in every pinned run
(the power cap, section 3). The median is the protocol's statistic; across
the last seven pinned runs (the six builds above and this one) the 8192 median
lies in 629.6 - 632.2 ms.

## 2. How it is measured

`tools/bench.py` follows `bench_prefill.py`'s protocol on our side:

- **Scope: forward only.** One CUDA-graph replay of the whole prefill, from
  token ids already on the device to the first token on the device:
  embedding, 64 layers, the closing RMSNorm, the lm_head row, the argmax.
  It is bracketed by `torch.cuda.synchronize()`. Weight loading, packing,
  kernel compilation, graph capture and tokenization happen before and are
  not timed. No scheduler, sampling server, detokenizer or IPC.
- **Repeats:** per length 2 untimed warmup replays, then 7 timed; the median
  is reported with min and max. Each repeat uses another prompt (the prompt
  rotated by one token), so no two repeats see the same ids (the first tokens
  of the 7 differ: `first_tokens` in the json).
- **Clock:** the SM clock is read with `nvidia-smi` right after each timed
  replay and recorded in the row; it is not used to correct anything.
- **Card:** the pinned card for the table; development numbers in
  `evidence/` come from GPUs 0 and 1 (same model, same 1500 MHz cap), which
  run about 1 % faster than the pinned card at 8192.
- `python run.py --prompt "..." [--prompt-tokens N]` prints the first token,
  the prefill time and tok/s with the same timing scope (prompts are padded to
  a multiple of 128 rows; causal, so the last real row is unchanged).

## 3. Which SGLang column, and why

**Column B (end to end), for all three lengths.**

- Column A (`one_batch`, forward only) has our scope. But at 8192 A is 9.1 %
  slower than B, and A's forward is a proper subset of B's work. Same code,
  same card: the two columns did not see the same machine state. A issues
  forwards back to back and records no clock; the card is power-capped (on
  this card a sustained GEMM shows `clocks_event_reasons.sw_power_cap` active
  and 1110-1470 MHz). A's 8192 number is most likely a throttled one: the
  baseline's three-card table shows a card at 1305 MHz finishing the same
  8192 work end to end in 696.07 ms, and A's 696.38 ms is within 0.05 % of
  it. That is consistent with A running near 1305 MHz; it is not a proof,
  because A recorded no clock. Our own runs on this card show the same
  mechanism at 8192: in each of the last six pinned runs the first timed
  replay (right after the two warmups, i.e. after three back-to-back 8192
  forwards) took 670.6 / 670.5 / 672.0 / 664.3 / 666.8 / 670.1 ms, 5.5-6.5 % above
  that run's median and the slowest of its 7, with the clock read after it at
  1365 / 1500 / 1380 / 1395 / 1500 / 1380 MHz
  (`reports/bench-final-v{9,8,7,6,5,3}-gpu2.json`, `all_ms` and `sm_clocks`); the
  later replays settle at 1440-1500 MHz. A sequence of forwards with no clock
  record can lose 6-9 % at this length.
- Column B records the SM clock right after its runs (1500 / 1485 / 1500
  MHz). Ours, read the same way, are 1500 / 1500 / 1440 MHz: B's clock is at
  least ours at 1024 and 8192; at 4096 ours is 15 MHz (1 %) higher, inside
  the 7-reading spread (1455-1500) and below the 2.4 % margin there. So B
  does not flatter us.
- B's fault is the service overhead it contains. The baseline shows it cannot
  just be subtracted (1024 would fall below its floor), so it overlaps the
  forward. Its visible part is the A-B gap where A is not throttled: 1024,
  2.1 ms. B is therefore lenient by about 2 ms at 1024 (we are 6.7 ms faster
  than A there as well), and strict at 4096 and 8192, where B is the faster
  column (by 2.8 and 58 ms).
- Picking per length is not allowed, and A at 8192 would hide a 9 % clock
  effect in our favour. B is the column whose errors are small and run against
  us at the two long lengths. Judged by B, this work is faster at all three
  lengths; judged by A, also at all three.

## 4. The FLOP count (`flops.py` checked)

`flops.py` counts 2 n P over every 2-D text weight, plus 2 n^2 d_attn for the
16 full-attention layers (causal halving included). Re-counted from the
safetensors headers (`tools/count_flops.py`):

- The `matmul` bucket (24.777 B) includes the **MTP layer** (`mtp.*`, 0.425 B
  2-D parameters), which a prefill that stops at the first token never runs.
  Text matmul weights without it: 24.351 B.
- Corrected totals: 50.08 / 202.78 / 412.15 TFLOP (flops.py: 50.9 / 206.3 /
  419.1), 1.6-1.7 % less. Scores are right: 0.21 / 3.30 / 13.19 TFLOP.
- Not counted by either: the lm_head row (2.5 GFLOP), our in_proj padding
  (16480 -> 16640 output columns, +1 % of in_proj), the delta rule's chunk
  algebra. The table keeps flops.py's numbers so that its TFLOP/s and % compare
  with the baseline's; the corrected ones are given beside them.

## 5. The 989 TFLOP/s denominator

The baseline divides by 989 x clk / 1980. A bf16 GEMM on this card (1500 MHz
cap) measured 759 TFLOP/s in a short burst, above 989 x 1500 / 1980 = 749:
impossible if 989 were the 1980 MHz figure. 989 TFLOP/s is 132 SMs x 4096
dense bf16 FLOP per clock x 1830 MHz. At 1500 MHz the peak is 811 TFLOP/s;
the table gives both denominators. Our sustained plain GEMM reaches 95 % of
132 x 4096 x clk in isolation (8192 x 16896 x 5120, K-slope 769 TFLOP/s at
1500 MHz); in the whole prefill the GEMMs run at 88-90 % of the at-clock peak.

## 6. Correctness

**The authored HIR (model.py) against the published model** (step 1, before
any kernel): `tilefoundry check` of all 18 functions of a linear-attention
layer (0) and a full-attention layer (3), plus `embed` and `final`, on the
model's own activations captured from `transformers` (`tools/hf_reference.py`),
at 1024, 4096 and 8192: **18 / 18 PASS at each length**
(transcripts not shipped; predicates in `tools/check_hir.sh`). The 64-layer
evaluator run gives the same first token as `transformers` (1024, 4096,
8192), max |logit error| 0.289 / 0.406 / 0.609, rel_l2 2.2e-2 / 1.6e-2 / 3.5e-2.

**The runtime twin against model.py** (`tools/check_twin.sh`, `tilefoundry
check runtime_model.py:Qwen3_8_27BTwin.<function>` with no `--expected`: the
reference is the evaluator on model.py): the same 18 functions at 1024, 4096
and 8192, **54 / 54 PASS** for the build of section 1 (`evidence/check_twin_v8/`,
the check transcripts (not shipped)); also 54 / 54 for each earlier build: this
round before STEP4 rows 43, 41, 39 and 36 (`evidence/check_twin_v7/`, `check_twin_v6/`, `check_twin_v5/`, `check_twin_v4/`), the previous alignment
round (`evidence/check_twin_v2/`), the one before any (`evidence/check_twin/`),
and this round's IEEE-exp trial (`evidence/check_twin_v3/`, section 7). Each bound is derived in
the script's header from where the twin rounds or sums in another order: one
bf16 landing can differ by one step = 2^-7 of the value (rtol 2^-7); residual
outputs get atol 2^-7 max|acc|; attention atol 2^-7 max|v| (the twin lands the
softmax weights in bf16 before the 1/l division, model.py after); delta_rule
keeps w, u, the masked q k^T and v_new in bf16 between and inside its kernels
(as FLA's production kernels do; model.py keeps f32): rel_l2 <= 2^-9 sqrt(5)
= 4.4e-3, atol 2^-6 max|out| = 0.073.

How these bounds came about, plainly: the first run used rtol 2^-8 for one
landing and atol 2^-8 max|out| for delta_rule; nine functions failed. Eight
failed by up to one bf16 step at the bottom of a binade -- that derivation was
wrong (two roundings of nearby values differ by one step, 2^-7, not half a
step), not the kernels. The ninth, delta_rule, failed because its first bound
counted only the output's landing and not the four bf16 intermediates; the
bound above counts them. It was set after the failure, so its margin is
stated: measured max |error| 0.03125 against 0.073 at all three lengths (one
bf16 step at the output's largest magnitude, 4.66; 0.0625 before the first
alignment round, whose phase 2b ran on tf32), rel_l2 2.01e-3 / 2.01e-3 / 2.04e-3
against 4.4e-3 (before: 2.23e-3 / 2.09e-3 / 2.11e-3); this round's delta-rule
kernels give the same output bit for bit (`tools/test_gdn.py`). For comparison, model.py's delta_rule is
1.8e-5 from transformers' (f32) one.

**First token against transformers, whole prefill on the twin**
(transcript not shipped; the build of section 1, same ids as the
reference; this round's earlier builds and the previous round's give the same
values, `results/e2e_twin_v{7,6,5,4,3}.txt`; the build before any TIR alignment:
`results/e2e_twin_final.txt`):

| prompt | twin first token | transformers | max abs logit error | rel_l2 | before alignment: max abs / rel_l2 | transformers' top-1 minus top-2 logit |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 641 | 641 | 0.2500 | 1.80e-2 | 0.1875 / 1.66e-2 | 6.00 |
| 4096 | 34594 | 34594 | 0.3750 | 2.14e-2 | 0.3438 / 1.26e-2 | 11.13 |
| 8192 | 16 | 16 | 0.8516 | 4.86e-2 | 1.5000 / 7.74e-2 | 11.25 |

Reading these: every function passes its own check at every length, and
changing nothing but attention's key-block size (same arithmetic, other
blocking) moved the 8192 result between 0.875 and 1.50 earlier (rel_l2
4.8e-2 to 7.7e-2). So the end-to-end number mostly measures how small
rounding differences grow through 64 layers; it says little about one kernel.
The twin's one systematic difference from model.py is the delta rule's bf16
intermediates (above), whose effect grows with the number of chunks. The first
token's lead over the second is 6 / 11 / 11 logits, at least 13 times the
largest error measured.

## 7. The four artifacts and how they correspond

| artifact | file(s) |
|---|---|
| authored HIR | `model.py` (`hf_alias()` maps the checkpoint's names) |
| scheduled HIR | `scheduled/{norm,embed_final,gemm,gdn_elementwise,gdn,attention}.py`: one Module per (model.py function, length), every model.py function covered |
| finalized TIR | `tir/*.py`, one file per (kernel function, length), from `tools/finalize_all.sh`: `tilefoundry schedule finalize` for each, run through `tools/finalize_deep.py` (the CLI's own `main` with a deeper Python stack, which the delta rule's `intra_wup` needs); one log per file in `evidence/finalize/` |
| tilelang runtime module | `runtime_model.py` (the twin of `model.py`, `@runtime_module` per HIR Module) + `kernels/` |
| entry point | `run.py` |

Which TIR comes from which scheduled HIR, and which scheduled HIR from which
part of model.py (`python tools/scheduled_index.py --table`; the same list
drives `tools/finalize_all.sh`; 75 TIR files, none missing):

| model.py function | scheduled HIR (module per length) | kernel @funcs -> tir/ | runtime kernels |
|---|---|---|---|
| `Qwen3_8_27B.embed` | `scheduled/embed_final.py`: `Embed{n}` | embed -> `tir/embed_*.py` | `elementwise.embed_kernel` |
| `Layer.input_norm / post_norm` | `scheduled/norm.py`: `RMSNorm{n}` | rmsnorm -> `tir/rmsnorm_*.py` | `elementwise.rmsnorm_kernel` |
| `GatedDeltaNet.in_proj` | `scheduled/gemm.py`: `InProj{n}` | pack (once), whole -> `tir/in_proj_*.py` | `pack.in_proj, gemm.gemm_kernel(plain)` |
| `GatedDeltaNet.prep` | `scheduled/gdn_elementwise.py`: `Prep{n}` | conv_q, conv_k, conv_v, gate_g, gate_beta -> `tir/prep_*.py` | `elementwise.prep_kernel` |
| `GatedDeltaNet.delta_rule (1 of 2)` | `scheduled/gdn.py`: `DeltaRuleIntra{n}` | intra_g, intra_wup -> `tir/delta_rule_*.py` | `gdn.intra_kernel` |
| `GatedDeltaNet.delta_rule (2 of 2)` | `scheduled/gdn.py`: `DeltaRuleRecur{n}` | recur_out -> `tir/delta_rule_*.py` | `gdn.recur_out_kernel` |
| `GatedDeltaNet.gated_norm` | `scheduled/gdn_elementwise.py`: `GatedNorm{n}` | gated_norm -> `tir/gated_norm_*.py` | `elementwise.gated_norm_kernel` |
| `GatedDeltaNet.out_proj (1 of 2 where split)` | `scheduled/gemm.py`: `OutProj{n}` | pieces (1024), whole -> `tir/out_proj_*.py` | `gemm.gemm_kernel(residual)` |
| `GatedDeltaNet.out_proj (2 of 2)` | `scheduled/gemm.py`: `OutProjTail{n}` | tail (1024) -> `tir/out_proj_*.py` | `gemm.gemm_tail_kernel(residual)` |
| `Attention.qkv (1 of 2 where split)` | `scheduled/gemm.py`: `QKV{n}` | pack (once), pieces (1024, 8192), whole, whole_b (8192), whole_last (1024, 8192) -> `tir/qkv_*.py` | `pack.qkv, gemm.gemm_kernel(qk_norm_rope)` |
| `Attention.qkv (2 of 2)` | `scheduled/gemm.py`: `QKVTail{n}` | tail (1024, 8192) -> `tir/qkv_*.py` | `gemm.gemm_tail_kernel(qk_norm_rope)` |
| `Attention.attention` | `scheduled/attention.py`: `Attention{n}` | attention -> `tir/attention_*.py` | `attention.attention_tir_kernel (attention_ws_kernel at n = 128)` |
| `Attention.o_proj (1 of 2 where split)` | `scheduled/gemm.py`: `OProj{n}` | pieces (1024), whole -> `tir/o_proj_*.py` | `gemm.gemm_kernel(residual)` |
| `Attention.o_proj (2 of 2)` | `scheduled/gemm.py`: `OProjTail{n}` | tail (1024) -> `tir/o_proj_*.py` | `gemm.gemm_tail_kernel(residual)` |
| `MLP.gate_up (1 of 2 where split)` | `scheduled/gemm.py`: `GateUp{n}` | pack (once), pieces (1024), whole -> `tir/gate_up_*.py` | `pack.gate_up, gemm.gemm_kernel(swiglu)` |
| `MLP.gate_up (2 of 2)` | `scheduled/gemm.py`: `GateUpTail{n}` | tail (1024) -> `tir/gate_up_*.py` | `gemm.gemm_tail_kernel(swiglu)` |
| `MLP.down (1 of 2 where split)` | `scheduled/gemm.py`: `Down{n}` | pieces (1024), whole -> `tir/down_*.py` | `gemm.gemm_kernel(residual)` |
| `MLP.down (2 of 2)` | `scheduled/gemm.py`: `DownTail{n}` | tail (1024) -> `tir/down_*.py` | `gemm.gemm_tail_kernel(residual)` |
| `Qwen3_8_27B.final` | `scheduled/embed_final.py`: `Final{n}` | final -> `tir/final_*.py` | `attention.final_kernel` |

Lengths in parentheses: the functions that exist at those lengths only (a
GEMM's split tail, section 8: 1024 for all but in_proj, 8192 for qkv;
qkv's `whole_b` at 8192, `whole_last` where the last M group has a tail);
"once": weight packing, lowered from the 1024 Module.

**How each link is held.**

- *model.py -> scheduled HIR.* Each scheduled Module's `entry` has the
  model.py function's parameters and calls the kernel functions. A call from
  one HIR function to another is a device call inside the same kernel
  invocation (`tilefoundry spec hir`, "Kernel invocation"), so an entry is
  one invocation. Where model.py's function needs two invocations it is two
  Modules called in order by the host: `delta_rule` (two execution contexts:
  (chunks, 48) CTAs x 128 threads, then (2, 48) x 384 -- "caller and callee
  MUST resolve one effective topology ... a mismatch is not a nested launch",
  same section), and every GEMM with a split tail (`<Name><n>` returns the
  whole tiles and the f32 pieces; `<Name>Tail<n>` on its own grid sums the
  pieces other CTAs wrote -- the grid-wide handover is the launch boundary).
- *scheduled HIR -> TIR.* `schedule finalize <file>:<Module>.<kernel func>`.
  Scheduled lowering takes one tensor result per function, so a model.py
  function with several outputs is either split into one device function per
  output on the same grid (`prep`: conv_q, conv_k, conv_v, gate_g, gate_beta,
  each selecting its CTAs with a mesh-coordinate range) or writes one packed
  tensor (`intra_wup`: [w | u | P row] per token and head; `recur_out` reads
  it as packed). Each TIR file starts with the tool's memory analysis.
- *TIR -> runtime.* Each runtime kernel implements one TIR function, or the
  device functions of one invocation in one launch (GEMM pieces + whole, for
  qkv also whole_b and whole_last; prep's five; intra_g + intra_wup), on the
  TIR's grid and thread count, with its ownership (which thread holds which
  element: e.g. the norms' (5, 256, 4) vectors, the GEMM tails' interleaved
  16-byte columns, the WGMMA fragment for decay and masks), its tiles,
  staging buffers, ring depths and shared-memory offsets, its instruction
  atoms (WGMMA shape, form and operand majors; TMA), its tile order, the
  index tables it reads (iota), and the warp(group)s it gives each role (e.g.
  the TMA store by warp 0 of the consumers). The backend adds what the
  tutorial says a backend adds (`tilefoundry tutorial schedule`, section 7):
  tensor maps and descriptors, mbarriers and phases, fences, WGMMA commit /
  wait, named barriers, `setmaxnreg`, and the warp shuffles that realize
  `T.reduce`.
- *check.* The scheduled HIR itself cannot be run by `check`: every Module
  here gives work to CTAs by mesh coordinate (persistent tile loops,
  per-CTA slices), and "evaluation MUST raise EvalError" on `MeshCoord`
  (`tilefoundry spec hir`, MeshCoord). The chain is checked where the
  tutorial puts the check: TIR -> backend -> `check` of the backend against
  model.py (section 6), which covers every function of the prefill.

**Where the runtime does something other than the TIR says.** The tutorial
(`tilefoundry tutorial schedule`, section 7): "TIR settles value ownership,
layouts, tile order, instruction atoms, and allocation. A backend
implementation preserves those decisions while supplying instruction encoding
and synchronization." The previous round listed eight differences. This round removed all of
them by changing the scheduled HIR, the runtime, or both; one more item
(output initialization) is the HIR's own form, which the tutorial tells
a backend it need not execute (below). Evidence: `evidence/STEP4.md` rows
27-43.

| was | now |
|---|---|
| GEMM tail: a device call in the HIR, a second launch in the runtime | the tail is its own Module (`<Name>Tail<n>`, its own grid: `Topology("cta", 128)`, 112 for qkv at 1024 -- the runtime's grid; the TIR said 132 before); the host invokes it after `<Name><n>`, as for delta_rule |
| attention, delta_rule: mask row / column numbers computed from thread coordinates | the kernels take the iota table the TIR reads (i32 in both HIRs now: an i64 table cost attention 45 us at 8192 in registers) |
| attention: V in TileLang's smem arrangement, 4 TMA boxes | V in the TIR's: five 16-key slabs of 4 x 64 columns (descriptor LBO 1024 elements, SBO 8 rows), five TMA copies, the value product as five m64n256k16 RS WGMMAs (P as five (128, 16) fragments) |
| attention: mask on every block in the TIR, on diagonal blocks in the runtime | the HIR masks only a block the diagonal crosses, under a guard loop of 0 or 1 iterations (`kb >= q0 // 80`) |
| attention: FA3's intra-warpgroup order in the runtime (S_kb issued ahead of block kb - 1's value product) | the runtime issues in the TIR's per-block order (S_kb, softmax_kb, O corr_kb, P_kb V_kb); only the two warpgroups' turn-taking at the tensor cores remains, which is synchronization. Bit-identical outputs; 1647.5 -> 1657.0 us per call at 8192, 107.6 -> 108.0 at 1024. Stating FA3's order in the HIR instead was tried (STEP4 row 30, `evidence/attention_fa3_hir_try/`): it needs a ring slot loaded in one iteration and read in the next, which the HIR cannot say without copying the tile, and finalize took the slot from the guard loop (always slot 0) |
| intra_wup: transposed blocks copied into smem scratch in the TIR, read in place by the runtime | the HIR reads them as bitcast views with swapped strides (no copy in the TIR) |
| intra: V's load and the P / w / u stores issued earlier than the TIR's statements (finalize puts a TMA load next to its first use, output stores at the end) | the runtime issues them where the TIR does and reuses bytes as the TIR does (X_u's second chunk over V's first, X_w's second chunk at xw1s -- since row 43 at its TIR address, over gb and K's first 7 KB --, w staged in V's place, u in its own tile, ost): 3 CTAs per SM kept; bit-identical outputs; 318.9 -> 321.5 us per call at 8192, 45.8 -> 49.3 at 1024. Without the TIR's reuse the TIR's issue points cost 414.7 us (K, V, both staging tiles and P alive together, 2 CTAs per SM) |
| intra's TIR itself could not be run as placed: three SW128 TMA tiles (ks, qs, vs) at 512 mod 1024 -- two 256-byte f32 rows before them shifted every later offset -- and four padded tiles (asm (64, 64) with row stride 65; maa, mbb, ys (32, 32) with stride 33) reaching past the bytes placement gave them (placement sizes a value by its shape, "logical bytes": `tilefoundry spec analysis`, ValueLifetime), so maa overlapped asm's last row and mbb overlapped maa while both were live | the HIR stages gc and beta as rows of one 1024-byte tile and states the padded shapes ((64, 65), (32, 33)), reading the first 64 / 32 columns; every SW128 tile now starts on a 1024-byte boundary and no tile reaches past its bytes (`tools/tir_placement_check.py` over all 75 TIR files, 467 placed tiles: 0 and 0; before: 27 and 12 in intra_wup). Why the alignment matters, measured on this H200 (`tools/tma_swizzle_probe.cu`, `results/tma_swizzle_probe.txt`): a 64 x 64 bf16 tile loaded by cp.async.bulk.tensor with SWIZZLE_128B at shared offset 512 lands by absolute-address swizzle, 0 of 4096 elements where the relative-swizzle layout `ComposedLayout(Swizzle(3, 4, 3), 0, layout)` of the TIR puts them (at offset 0: 4096 of 4096). The runtime clears the padded tiles as the TIR does (+0.8 us per call at 1024, +5.1 at 8192) |
| shared-memory addresses: TileLang's | every kernel at the TIR's byte offsets (`python tools/smem_plan.py --check`: 0 differences at 1024, 4096, 8192, `results/smem_plan_check_v3.txt`). GEMM main and tail, attention, prep, final, recur_out: `tl.disable_shared_memory_reuse` gives each buffer its own region in name order, the buffers are named in the TIR's address order (`sm<k>_<tir name>`), and where the TIR reuses an offset the kernel reuses the buffer. intra, whose TIR plan lets tiles of different dtype and size share bytes in turn, overlapping in part: one (584, 64) bf16 allocation (74752 B, the TIR's extent); each bf16 tile a row slice of it at its TIR offset (all multiples of 1024 B = 8 rows, so TileLang's SW128 swizzle of the allocation is each tile's), each f32 tile (gb, asm, maa, mbb, Y, gsm) read and written at its TIR byte address in the same allocation (`kernels/gdn.py` `OFF`, `tf_ld` / `tf_st`), a barrier between a tile's last read and the next tile's first write on its bytes. Bit-identical outputs; 3 CTAs per SM as before; isolated (CUDA events, 5 x 20 calls) 49.8 -> 49.0 us at 1024, 170.6 -> 166.2 at 4096, 327.6 -> 319.6 at 8192 |
| qkv: three band loops per M group (the TIR held three TMA rings, 496 KB of smem, beyond the 227 KB an SM has) | one band loop per group with q- and k-head guards (one ring, 208 KB, the runtime's); the runtime's epilogue has the same two guards, its tail kernel the TIR's k-only guard |
| recur_out: the TIR copied the loop-carried bf16 state (`sb = sb_next`) into a 16 KB buffer every chunk, and shared the output tile's offset with `vd`, which the two warpgroups use at the same time | the state's bf16 copy is made at the top of each iteration (no copy, one buffer less); the output tile 2-deep, as the runtime holds it |

What remains: nothing. The last item, intra's shared-memory byte offsets,
is closed (row above; STEP4 row 43). It took four tries; the three that
TileLang 0.1.15 refused were typed views of one allocation at an offset:
`T.view` needs a zero offset and an equal bit count; a typed view from an
address (`T.make_tensor_from_addr`) fails in MakePackedAPI ("variables (As,
Bs) are used, but are not passed in as API arguments"); a buffer aliasing
the allocation at an element offset (`T.Buffer(..., data=raw.data,
elem_offset=k)`) fails in layout inference ("InputShape() = (64, 128) shape =
(20480,)"). What works keeps one buffer object: row slices of it for every
bf16 tile (TMA and WGMMA take slices) and byte addresses into it for the f32
tiles, which only plain loads and stores touch. Placing each buffer in a
region of its own would have needed 118.7 KB (1 CTA per SM instead of 3).

Not differences, though they look like ones. *Output initialization:*
`tf.zeros` + `insert_slice` lowers to `T.fill(out, 0)` (and a kernel
function that takes the output as a parameter lowers to `T.copy(out, out_1)`,
e.g. qkv's `whole_last`). That is how the HIR has to say "a tensor built tile
by tile": a loop-carried value needs its first-iteration value from before
the loop (`tilefoundry spec hir`, LoopRegion, "Carry-out semantics"), and
`Zeros` is the HIR's only allocating op (`tilefoundry spec dsl`, the op
list). The tutorial says what a backend does with it: "a CUDA implementation
need not clear the whole output before overwriting every tile" (`tilefoundry
tutorial schedule`, section 6). The runtime writes every tile in place
(clearing first would write every output twice: gate_up's alone is 285 MB
per layer at 8192). Exp and division in SiLU (gate_up's
epilogue, prep, gated_norm) are encoded as `__expf` / `__fdividef` (the
SFU's ex2 / rcp approximations); the TIR says Unary EXP and Binary DIV and
leaves the encoding to the backend, and `check` holds the result to model.py
(54 / 54, section 6). The IEEE encodings were tried this round: per call at
8192 gate_up 3980 -> 4127 us, prep 156 -> 220, gated_norm 77 -> 101; the
whole prefill 644.00 ms on the pinned card (`results/bench-v4-ieee-exp-gpu2.json`),
slower than SGLang. `T.reduce` as a warp butterfly
plus a workspace slot per warp is the runtime's reduce (`tilefoundry spec
runtime 2.6.6`: the tier follows from the layouts, a mesh crossing warps
needs the workspace); mbarriers, named barriers (the FA3 turn-taking of the
two attention warpgroups), WGMMA fences and commit / wait, `setmaxnreg` and
descriptors are the synchronization and encoding the backend supplies.

**The analysis header's `error=` lines.** Each TIR file starts with the
tool's memory analysis; some carry two kinds of errors:
- *rmem placement peak* above 256 KB (attention 320 KB, final 960 KB, qkv
  256.5 KB): the sum of the register tensors the TIR keeps live; it
  materializes, e.g., final's 32 x 5120 f32 product before the reduction that
  consumes it. ptxas allocates the registers of the compiled kernel; every
  runtime kernel compiles without spills (`results/ptxas_v9.txt`: registers
  per thread and spill bytes per kernel and length).
- *l2 reuse window ... exceeding capacity*: the cache model's estimate of
  the data a loop re-reads (e.g. the GEMMs' band loop); a performance note on
  the raster, not a placement. The raster was chosen by measurement (STEP4
  row 15).
No smem placement error remains (qkv's 496 KB was the only one).

**What scheduled lowering accepted on this build** (each found by
`schedule candidates` / `finalize` refusing something; the HIR was changed to
fit, and the runtime followed):
one tensor result per lowered function; no lowering for `Arange`,
`FullLike`, `RepeatInterleave`, `Softplus`, `Silu`, `RMSNorm` (written out
in their arithmetic); a binary op's sharded inputs must be on one mesh (a
gamma split over threads only cannot multiply x split over CTAs and threads;
mesh-coordinate slicing, as above, avoids that); operands of a binary op never
promote and an integer literal is i64 (`tf.cast(0, "i32")` is refused at
lowering as an unscheduled cast; an i32 zero is `tf.zeros(..., "rmem")`);
`Reduce` wants its source with explicit strides; bool tensors cannot be
copied; loop bounds may depend on mesh coordinates, but a bound must be a
bounded form: a loop variable or `kb * BN` alone is refused ("no bounded
span"), `(cta.x * 0 + kb) % N` is accepted, and a step must be a literal;
WGMMA operand arrangements per `schedule facts` (repeat modes leading, the
MN-major chunk stride above); a transpose of a sub-mesh register tensor fails
TIR verification, and `tf.transpose` of an smem tile is copied into smem
scratch, while a `tf.bitcast` view with swapped strides (offset included) is
read in place; the delta rule's long update chain needs a deeper Python stack
than the CLI's (`tools/finalize_deep.py`). finalize orders statements itself:
it places a TMA load next to its first use and output stores at the end,
whatever the HIR's order. Finalize takes seconds to tens of minutes (the
memory analysis over isl relations; attention the longest); it grew past an
hour whenever one function kept several TMA rings live together (qkv's band
loops inside the group loop) or held qkv's epilogue over 3 M groups; qkv now
has one band loop per group with guards, and 8192's `whole` is split into
`whole` (groups 0-1) and `whole_b` (group 2), device calls of one launch.

## 8. Segmentation and fusion: the plan and why

Evidence first, plan second (`evidence/STEP3.md`: step 2 per-length analysis,
where `analyze` and the hand count disagree and which is wrong, and the step-3
log; `evidence/STEP4.md`: every kernel change with its numbers).

**Fusion: one set of kernel boundaries for all three lengths** (each change
removes traffic that grows with the prompt and trades against nothing):

| kernel | fuses | evidence |
|---|---|---|
| `in_proj` | the four projections as one GEMM over a packed weight | b and a alone are N = 48 GEMMs |
| `prep` | conv + SiLU + q/k l2norm + g + beta | gmem per layer 8.44 -> 1.66 GB at 8192 (STEP3 row 7) |
| `delta_rule` | two kernels: `intra` (chunk-parallel) + `recur_out` (state and output in one serial pass) | predicted 5326 -> 3960 us per layer at 8192 (rows 4-5); BV 64 (row 6) |
| `gated_norm` | alone | needs a head's 128 value columns, which BV 64 splits |
| `out_proj`, `o_proj`, `down` | GEMM + residual add | row 10 |
| `qkv` | GEMM + per-head q/k RMSNorm + partial RoPE | a 256-wide tile is one head |
| `attention` | flash + causal skip + sigmoid gate | gmem 67.4 -> 12.4 GB at 8192 (row 1) |
| `gate_up` | GEMM + SwiGLU | 1.36 GB per layer at 8192 -> 0 (row 9) |
| `input_norm`, `post_norm` | alone | a row's norm needs all 5120 columns; a GEMM tile has 256 |

**Segmentation: the GEMM decomposition depends on the length; everything
else is one configuration.** The rule (`kernels/gemm.py` `plan`): 128 x 256
tiles; GROUP = 16 M-tiles where M has 16k tiles, else 8; the last R = T mod
132 tiles (rounded to whole bands) are split along K into s pieces when there
are whole tiles and s >= 2 fits (R s <= 132, s | K-blocks). For the three
benchmark lengths:

| | 1024 | 4096 | 8192 |
|---|---|---|---|
| GROUP | 8 | 16 | 16 |
| split tail | gate_up 32 x 4, down / out / o_proj 32 x 4, qkv 56 x 2 | none | qkv 32 x 4 |
| attention | `attention_tir` (the TIR's per-block order, the two warpgroups in FA3's ping-pong), 128-row query block, BN 80 | same | same |

**Where the boundary is, measured.** STEP3 had written "two segments, cut
between 1024 and 4096" and left the place of the cut to measurement. The
measurement (`tools/segment_study.sh`, the whole prefill by the bench
protocol, 2 + 7, GPU 1, one session, the GEMM decomposition forced to each
configuration; `reports/segment2/*.json`; the median, range in brackets):

| n | M-tiles | g8 split | g8 no split | g16 split | g16 no split | rule (shipped) | rule vs best forced |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 8 | 79.97 (79.7-80.3) | 94.54 (94.1-94.7) | = g8 | = g8 | 79.89 (79.5-80.1) | -0.1 % |
| 1536 | 12 | 118.09 (117.7-118.3) | 120.34 (120.3-121.0) | = g8 | = g8 | 117.90 (117.5-121.4) | -0.2 % |
| 2048 | 16 | 155.50 (154.2-156.4) | 162.79 (162.6-163.5) | 154.61 (154.3-158.2) | 162.32 (162.1-162.5) | 154.24 (153.9-160.5) | -0.2 % |
| 2560 | 20 | 195.12 (194.0-196.1) | 206.44 (206.3-211.2) | = g8 | = g8 | 194.94 (194.1-200.6) | -0.1 % |
| 3072 | 24 | 233.43 (233.0-244.5) | 233.55 (232.3-241.2) | = g8 | = g8 | 233.03 (232.6-244.8) | -0.2 % |
| 4096 | 32 | 306.46 (305.0-312.5) | 306.35 (305.6-315.5) | 305.22 (303.6-315.6) | 304.78 (303.6-311.8) | 304.93 (303.6-319.3) | +0.1 % |
| 8192 | 64 | 619.67 (619.0-656.3) | 620.09 (619.7-664.4) | 619.89 (617.5-657.7) | 621.59 (618.6-658.7) | 620.85 (616.8-657.0) | +0.2 % |

("= g8": GROUP 16 does not divide the M-tiles, so the forced 16 falls back to
8 and the program is the same.) SM clock 1470-1500 MHz in every cell. A first
run of this study (not shipped) is discarded: another user's process
came onto GPU 1 during its "g16 split" configuration and moved those cells by
+15 to +28 %; the per-run process list in the JSON shows it.

Repeated on this round's build (`reports/segment3/`, STEP4 row 38, same
protocol and card; the later rows 39-43 change only delta_rule's intra
kernel, no GEMM and no segment boundary; no other process on the card in
any cell, SM clock 1470-1500 MHz):

| n | g8 split | g8 no split | g16 split | g16 no split | rule (shipped) | rule vs best forced | split saves (g8) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 80.16 (80.0-80.3) | 94.63 (94.2-95.1) | = g8 (79.95) | = g8 (94.66) | 79.95 (79.5-80.1) | -0.0 % | 15.3 % |
| 1536 | 118.08 (117.9-118.7) | 120.26 (120.0-120.8) | = g8 (118.17) | = g8 (120.18) | 118.06 (117.8-118.6) | -0.0 % | 1.8 % |
| 2048 | 155.60 (154.9-156.1) | 162.75 (162.6-163.6) | 154.34 (154.0-155.7) | 162.39 (162.3-162.5) | 154.83 (153.8-157.2) | +0.3 % | 4.4 % |
| 2560 | 194.90 (194.2-195.4) | 206.38 (206.3-206.9) | = g8 (194.91) | = g8 (206.43) | 195.02 (194.1-226.6) | +0.1 % | 5.6 % |
| 3072 | 232.59 (232.3-263.4) | 233.22 (232.3-245.6) | = g8 (233.48) | = g8 (233.56) | 233.08 (232.9-244.1) | +0.2 % | 0.3 % |
| 4096 | 306.16 (305.2-312.8) | 306.60 (305.5-312.0) | 304.90 (304.2-310.2) | 305.01 (303.9-311.3) | 304.99 (304.6-309.3) | +0.0 % | 0.1 % |
| 8192 | 620.20 (618.6-657.8) | 620.34 (618.9-651.8) | 618.59 (617.1-655.5) | 618.07 (617.2-652.1) | 618.80 (615.2-654.3) | +0.1 % | 0.0 % |

Every cell is within 0.6 % of the first study's, and so is every conclusion
below. (The study runs on GPU 1, about 1.8 % faster at 8192 than the pinned
card of section 1.)

What the numbers say:

1. **The split is the only length-dependent choice that matters**: 15.4 % at
   1024, 1.9 % at 1536, 4.5 % at 2048, 5.5 % at 2560, nothing at 3072, 4096
   and 8192. GROUP 8 vs 16 is inside the run-to-run range at every length
   where both exist (2048: -0.6 %, 4096: -0.4 %, 8192: +0.0 %, split on).
2. **The boundary is not a length.** 2560 gains more than 2048, 1536 less
   than both, 3072 nothing. What decides is the tail wave of each GEMM,
   T mod 132 for T = (n / 128)(N / 256) tiles. A wave model with one rate
   measured elsewhere (`tools/segment_model.py`: t(K) per tile from gate_up at
   4096; tail = one wave of t(K) / s plus the 6.6 us sum kernel) predicts the
   saving per length:

   | n | GEMMs the rule splits (R tiles x s pieces, GROUP 8) | predicted ms | measured ms |
   |---:|---|---:|---:|
   | 1024 | gate_up 32x4, down 32x4, out_proj 32x4, qkv 56x2, o_proj 32x4 | 16.6 | 14.58 |
   | 1536 | gate_up 48x2, qkv 12x8 | 2.5 | 2.26 |
   | 2048 | gate_up 64x2, down 56x2, out_proj 56x2, o_proj 56x2 | 10.4 | 7.29 |
   | 2560 | down 20x4, out_proj 20x6, o_proj 20x6 | 13.9 | 11.32 |
   | 3072 | qkv 24x5 | 0.7 | 0.12 |
   | 4096 | - | 0.0 | -0.11 |
   | 8192 | qkv 24x5 | 0.7 | 0.42 |

   The model gets the order and the zeros right and over-predicts by 10-40 %
   (the split wave's pieces are short, so their prologue and the sum kernel
   weigh more than the model's one constant).
3. **So the segmentation is a per-GEMM rule, not two length segments.** The
   rule in `plan` is evaluated per GEMM shape at each length, and the
   measured "rule" column is within 0.2 % of the best forced configuration at
   all seven lengths. The earlier wording "1024 vs 4096/8192" was the
   hypothesis; the measured boundary for the N = 5120 GEMMs (down, out_proj,
   o_proj), which carry most of the saving, is: split while the tail is
   <= 66 tiles after rounding to bands, which holds at 1024, 2048, 2560
   and fails at 1536 (108), 3072 (84 -> 88), 4096 (112), 8192 (92 -> 96).

Attention is one configuration at all lengths, measured at each
(`tools/test_attention.py n variant`, isolated, 10 calls, GPU 1):

| attention | 1024 | 4096 | 8192 |
|---|---:|---:|---:|
| BM 64 (STEP4 row 6, before) | 143 us | 876 us | 2821 us |
| BM 128, warp-specialized (`attention_ws`, BN 112) | 106.8 | 578.6 | 1722.1 |
| BM 128, FA3 ping-pong (`attention_pp`, BN 80) | 104.7 | 554.5 | 1633.1 |
| the same with the TIR's V slabs and iota table (section 7) | 108.4 | 569.9 | 1659.0 |
| ... and the TIR's per-block issue order, ping-pong only between warpgroups (`attention_tir`, shipped) | 108.7 | 565.9 | 1655.2 |

At 1024 the 24 heads x 8 query blocks give 192 CTAs (1.45 waves); BM 64
doubles the CTAs and is still slower there (143 vs 105 us), so attention gets
no 1024 segment. It is 1.7 ms of 80 ms at 1024.

Alternatives measured, kept and not kept (numbers per length where the change
applies; full rows in STEP4):

| decision | alternative | numbers |
|---|---|---|
| split tail at 1024 | whole tiles only | 1024: GEMMs 86.29 -> 74.51 ms per prefill, wall 103.62 -> 91.89 ms (row 1) |
| tail split along K, reduced by a 2nd kernel | narrower tail tiles along N (no reduction) | per-tile efficiency at 8192: BN 256 766 TF/s, 160 611, 128 520, 64 290 (row 8b) |
| packed weights | weights read in place, gate_up as two accumulators | gate_up 1024: 556 -> 598 us; 8192: ~4100 -> 4353 us (row 8a) |
| GROUP 16 at 4096 / 8192 | 8, 32 | sustained 8192: gate_up 4184 / 4153 / 4205 us for 8 / 16 / 32 (row 15) |
| epilogue on the consumers | on separate warps (handoff through smem) | no gain: plain 1844 -> 1890, swiglu 4064 -> 4076 us (row 7a) |
| residual add in smem | R prefetched by the producer; R into registers | 8192 K=6144: 778 vs 785 vs 800 us (row 16) |
| attention FA3 ping-pong, BM 128, BN 80 | BM 64; WS BN 64 / 112; FA3 intra-WG overlap | 8192, isolated: 2764 (BM 64) -> 1735 (WS, BN 112) -> 1633 us (ping-pong, BN 80); intra-WG overlap 2411 (rows 6, 6a, 9) |
| intra per value head | per key head (3 heads per CTA) | 8192: 453 -> 298.6 us (row 11) |
| delta-rule cut P2 (2 kernels) | P1 (3 kernels), v0 | STEP3 rows 4-5 |

## 9. Where the time goes (final, pinned card, nsys, one prefill)

`results/nsys/v10_gpu2.table.txt` (the build of section 1, CUDA graph traced
per node; nsys adds 2-4 % to the wall time; the build before STEP4 rows 41
and 43: `v9_gpu2.table.txt`, intra 16.67 ms at 8192; the previous round's
build: `final_gpu2.table.txt`):

| kernel | calls | 1024 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| gate_up (GEMM + SwiGLU) | 64 | 33.11 ms | 136.91 ms | 269.96 ms |
| down / out_proj / o_proj (GEMM + residual) | 128 | 22.63 | 93.60 | 184.12 |
| in_proj (GEMM) | 48 | 11.21 | 47.89 | 94.26 |
| qkv (GEMM + norm + RoPE) | 16 | 3.51 | 14.75 | 28.25 |
| split-tail sums | 208 / 0 / 16 | 1.43 | - | 0.11 |
| attention | 16 | 1.82 | 9.73 | 28.28 |
| delta rule intra (intra_g + intra_wup) | 48 | 2.15 | 8.64 | 16.53 |
| delta rule recur_out | 48 | 2.00 | 7.94 | 15.13 |
| prep | 48 | 1.31 | 4.19 | 7.97 |
| rmsnorm | 128 | 0.76 | 2.78 | 5.30 |
| gated_norm | 48 | 0.54 | 2.02 | 3.89 |
| final (norm + lm_head) | 1 | 0.57 | 0.57 | 0.57 |
| sum of kernels | | 81.1 ms | 329.1 ms | 654.4 ms |

The GEMMs are 88-89 % of the time at every length and run at 88-92 % of
the peak at the clock the power cap allows; the gap to the at-clock floor
(529 ms at 8192) is mostly theirs. Of the rest, the delta-rule recurrence is
latency-bound (2.4 us per 64-token chunk on 96 CTAs: a direct probe says 0.74
us of it is the v_new GEMM's sync chain, 0.42 us the bf16 copy of the state).

## 10. Process notes

- Order kept: model.py checked against `transformers` first; `analyze` at
  each length; the cut decided from that evidence; then schedules and
  kernels, one logged change at a time (`evidence/STEP3.md`, `STEP4.md`).
  Step 2 redone on the final model.py per length, beside the hand count and
  the measured kernels: `evidence/analyze/v1/VS_MEASURED.md` (STEP3, second
  part). Rerun this round (`tools/analyze_all.sh model.py v2`, `--compute-cost
  --memory --roofline --performance --dim seq_len=N`, 14 functions x 3
  lengths, `results/analyze_v2.log`): `evidence/analyze/v2/TABLE.md` is
  identical to v1's -- e.g. the whole prefill's roofline sum 95.2 / 417.8 /
  942.0 ms, of it compute-bound 50.4 / 201.7 / 403.3 ms, and the per-function
  disagreements with the hand count (STEP3, "Where `analyze` says memory and
  the hand count says compute") stand as written.
- The last round wrote scheduled HIR for every function that had none
  (norms, embed, final, prep, gated_norm, qkv, attention, delta rule) and made
  each kernel follow its finalized TIR. What that changed, measured on the
  same card before and after (STEP4 rows 17-26):

  | change to follow the TIR | 1024 | 4096 | 8192 |
  |---|---:|---:|---:|
  | rmsnorm: (5, 256, 4) vectors, both rows per thread | 6.8 -> 6.0 us | 23.2 -> 22.0 | 43.4 -> 41.7 |
  | prep: (67, 128) smem slab, four tap reads | 22.4 -> 27.9 us | 82.9 -> 87.8 | 158.1 -> 165.7 |
  | GEMM tails: 384 threads, interleaved 16-byte ownership | 6.6 -> 6.5 us | - | 6.8 -> 6.5 |
  | GEMM ring: stage k mod 3 restarting per tile | gate_up 3953.9 -> 3953.9 us (8192, isolated) | | |
  | gate_up epilogue: bf16(acc) staged whole, 16 x 16 grid | 509.9 -> 513.7 us | 2140.9 -> 2172.9 | 4190.4 -> 4242.6 |
  | qkv epilogue / tail: 16 x 16 grid, interleaved tail | 224.4 -> 224.2 us | 936.6 -> 941.8 | 1768.5 -> 1781.2 |
  | delta rule: intra_g + intra_wup (packed w / u / P), f32 phase 2b | 42.8 -> 42.8 us | 166.9 -> 171.0 | 316.5 -> 326.3 |
  | whole prefill (bench, pinned card) | 80.46 -> 80.48 ms | 307.62 -> 310.05 | 629.61 -> 632.16 |

  (per-call kernel times from `results/nsys/final_gpu2.table.txt` vs
  `v6_gpu2.table.txt`; nsys runs vary by about 1 % at 4096.)
- This round removed the remaining differences between runtime and TIR
  (section 7; STEP4 rows 27-43). Per call, nsys on the pinned card,
  previous round's build -> this one (`final_gpu2.table.txt` ->
  `v9_gpu2.table.txt`; rows 41 and 43: `v9_gpu2` -> `v10_gpu2.table.txt`):

  | change | 1024 | 4096 | 8192 |
  |---|---:|---:|---:|
  | GEMM tails as Modules of their own (HIR only) | - | - | - |
  | qkv: one band loop per group with q / k guards (TIR smem 496 -> 208 KB); runtime epilogue the same guards | 224.2 -> 222.2 us | 936.6 -> 927.1 | 1768.5 -> 1768.8 |
  | attention: V in the TIR's 16-key slabs (five RS WGMMAs), iota table (i32), diagonal mask stated in the HIR, the TIR's per-block issue order | 111.1 -> 115.1 us | 599.6 -> 611.6 | 1735.1 -> 1773.8 |
  | intra: transposes as bitcast views, gc / beta / g staged, iota table (i32), u then w, the TIR's copy issue points and buffer reuse | 42.8 -> 46.0 us | 166.9 -> 182.0 | 316.5 -> 347.2 |
  | recur_out: S's bf16 copy at the loop top, output tile 2-deep | 42.6 -> 42.3 us | 160.0 -> 166.4 | 305.5 -> 311.0 |
  | shared memory at the TIR's offsets (all but intra) | gate_up 509.9 -> 507.9 us | 2140.9 -> 2140.1 | 4190.4 -> 4224.3 |
| intra: the TIR's placement made executable (row 41: gb one 1024-byte tile, padded shapes cleared), then every tile at its TIR byte offset in one allocation (row 43) | 46.0 -> 44.8 us | 182.0 -> 179.9 | 347.2 -> 344.3 |
  | tried, not kept: IEEE exp / division in SiLU | prefill 82.73 ms | 315.71 | 644.00 |
  | intra with the TIR's copy issue points: without the TIR's buffer reuse (not kept) / with it (kept) | 45.8 -> 59.7 / 49.3 us (isolated) | | 318.9 -> 414.7 / 321.5 |
  | whole prefill (bench, pinned card) | 80.48 -> 80.29 ms | 310.05 -> 309.61 | 632.16 -> 631.19 |

  Kernel times in an nsys run move by 1-2 % between runs (4096 above), so
  the GEMM rows' changes are noise; attention and intra are the costs of
  this round (0.6 + 1.3 ms at 8192, intra 316.5 -> 344.3 us per call),
  recovered elsewhere in the run-to-run spread of the whole prefill.
- TileLang 0.1.15 findings, confirmed / refuted notes and new facts:
  the round archive.
- Reproduce: `tools/check_hir.sh`, `tools/check_twin.sh N "0 3"`,
  `tools/finalize_all.sh`, `tools/smem_plan.py --check`, `tools/ptxas_report.py`,
  `tools/e2e_twin.py --lens 1024,4096,8192 --graph`,
  `CUDA_VISIBLE_DEVICES=2 tools/gpurun.sh python tools/bench.py`.
