# Qwen3.8-27B prefill at 128–1024 tokens — across the roofline ridge

A 64-layer gated-delta-net / attention hybrid, prefill only: token ids in, first
token out. Batch 1, no history, **128 / 256 / 512 / 1024 tokens**, bf16, no
quantization, no vision, no decode. Produced by one agent against `main` at
`18479169`, 2026-10-15.

```
19.57 / 26.37 / 44.02 / 82.66 ms    SGLang 0.5.21: 36.84 / 37.16 / 48.54 / 85.41
1.88x / 1.41x / 1.10x / 1.03x       same card, same weights, same scope, interleaved
322 / 478 / 573 / 611 TFLOP/s       33% / 48% / 58% / 62% of the H200's 989.5
first token identical to transformers and to SGLang at all four lengths
```

Those four rows are the state at which the task's criteria were met. The kernels
here are one optimization pass newer and faster per call, with no end-to-end
number of their own — section 6 says exactly what was and was not re-measured.

**This one straddles the roofline ridge.** A bf16 weight GEMM with `T` rows does
`T` FLOP per byte of weight; the H200's ridge is 989.5 / 4.8 = 206 FLOP/byte. So
at 128 tokens the prefill is **memory bound** — its floor is the 51.4 GB of
weights at 4.8 TB/s, 10.71 ms — and from 256 on it is **compute bound**. The
lower bound changes kind inside the range this example covers, and the answer to
"can these four lengths share one implementation" has a physical reason rather
than a taste.

The answer was no: **four lengths, four implementation sets, and all 64 lowered
TIR files are distinct.** Serving 128 out of the 256-row implementation costs
+4% on the GEMM that was measured in isolation but **+35.6% on the forward**, and
one implementation fewer only saves compile time and 0.21 GB of static buffers.
The evidence is the cross-plan matrix in [`BENCH.md`](BENCH.md) section 7.

The full argument for every number is in [`BENCH.md`](BENCH.md), which is the
run's own report kept whole. This file is the front door.

---

## 1. Environment

Nothing here is installed by this directory; this is what it was written and
measured against.

| | |
|---|---|
| GPU | NVIDIA H200 SXM (143 GB), **SM clock 1500 MHz** under load |
| CUDA | 13.2, `CUDA_HOME` must be set |
| Python | 3.12 |
| `tilefoundry` | `0.0.2+35.g18479169`, from the wheel, **not** editable — nothing here reads the source tree |
| from the environment | `torch` 2.14.1+cu130, `transformers` 5.14.1, `safetensors` 0.8.0, `tokenizers` |
| extra package | `tilelang` 0.1.15 |
| weights | the published `Qwen3.8-27B`, bf16 |
| the opponent | `sglang` 0.5.21 (`torch` 2.13.0), in its own virtualenv — **not** this one |

**The clock is 1500 MHz, not the 1980 the card reports as its maximum.** Locking
it needs a permission this account does not have; the card sits at 1500 under
load, and `ncu` reads 1.42–1.46 GHz inside a tensor-core kernel even then. Both
benchmark workers read the SM clock after every timed iteration and put the range
in the row, so a run at another clock is visible rather than silently comparable.
989.5 TFLOP/s assumes the 1.83 GHz boost; at 1500 MHz the same product is 811,
and `BENCH.md` reports both denominators.

### One directory you have to name, one it writes itself

```bash
export QWEN38_CKPT=/path/to/Qwen3.8-27B        # the published checkpoint
python tools/prepare_ckpt.py                   # writes ./ckpt once, about 54 GB
```

There is deliberately no default for `QWEN38_CKPT`: a path that exists on one
machine is not a setting. `paths.py` is the only place this is decided, and the
scripts that need it also take `--ckpt`. The benchmark needs one more, for the
same reason: `$SGLANG_PYTHON`, the python of a virtualenv with SGLang in it
(section 6). Neither has a fallback, and `$TILEFOUNDRY` / `$PYTHON` override
which CLI and which interpreter the scripts shell out to (default: whatever is
on `PATH`).

`./ckpt` is the same weights under the canonical names `model.py` declares,
written once by `Module.prepare`; the runtime reads it and nothing else. It is
not shipped and not a setting.

## 2. How this was produced

One agent, working in a directory with nothing of ours above it.

| | |
|---|---|
| agent | Claude Code, `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint, the backend (tilelang), the deliverable filenames, two free H200s, an installed SGLang in its own virtualenv, and a page of TileLang gotchas from another loop on the same card |
| told | everything about TileFoundry is to be asked of the `tilefoundry` command; the model itself is its own to research; **measure SGLang yourself** — no baseline table was given |
| `tilefoundry` | read-only in its venv, from the wheel built at `18479169`; verified afterwards byte for byte unmodified (341 files) |
| human input | two turns: the prompt, then one sentence (both below) |
| duration | 12.28 h, 1735 tool calls (586 on the main line, 1149 in 5 sub-agents), 1 context compaction |

**The baseline was not handed over this time.** The previous prefill example
([`qwen3_8_27b_prefill-tilelang`](../qwen3_8_27b_prefill-tilelang)) gave the run
a measured SGLang table and asked only for its own half. Here it had to build
the instrument as well, and these lengths make that harder, not easier: a
serving path's fixed cost is tens of milliseconds, which at 128 tokens is larger
than the forward being measured. What the run was told about it was two
failure modes and nothing else — that a repeated prompt may not recompute, and
that a shape's first run is not the same number as its later ones. The protocol
it built from that is `BENCH.md` sections 1–3: two workers driven by one process
on one card, alternating round by round, each iteration asserting that it really
recomputed (NaN-poisoned K/V slots and GDN state on the SGLang side, every static
buffer on its own side), with a stubbed forward as the negative control that
makes the assertion fail.

<details>
<summary><b>The prompt, in full</b> (translated from the Chinese it was written in)</summary>

> Get Qwen3.8-27B's **prefill** right on TileFoundry, and make it fast.
> Weights + config: `<checkpoint>`. Backend: tilelang.
>
> Text prefill main path only: **batch = 1, history = 0**, prompt lengths
> **128 / 256 / 512 / 1024**. No decode (stop after the first token), no vision
> tower, no quantization.
>
> Everything about TileFoundry is to be asked of the `tilefoundry` command — do
> not ask people, do not look elsewhere. The model itself is yours to research.
> **Do not go looking for a ready-made implementation**: nothing on the web, in
> another checkout or in another examples directory is this round's source of
> answers. Walking it yourself is the point.
>
> **The opponent is SGLang, and this time you measure it yourself.** It is
> installed at `./sglang-env/bin/python` (0.5.21 / torch 2.13) — a different
> environment from this round's `.venv`. Measure it and measure yourself, **both
> sides counting the forward only**: weight loading, compilation, tokenization
> and the serving path's fixed costs count on neither side. Note that this
> round's lengths are short — that fixed cost is of the same order as one
> forward, or larger, so "my milliseconds are smaller than its" is free when the
> scope is wrong. Write down what your timer covers, from where to where, and
> why that cost is not inside it.
>
> How to measure is your judgement. Two things will destroy the numbers
> outright; go and confirm whether you stepped on them: a repeated prompt may
> **not recompute** the second time — you must be able to produce evidence, an
> assertion and not a spot check; and a shape's **first** run is not the same
> number as the ones after it — write the warm-up and sampling rules into the
> protocol. Report the range beside the median, and read the SM clock on every
> iteration.
>
> **Count the FLOPs yourself**, from the safetensors headers; do not trust the
> parameter count on the model card. Your TFLOP/s and your roofline bound both
> come out of that ledger, and the ledger itself is an artifact.
>
> **Two cards — but both numbers in a row must come from the same one.**
> `CUDA_VISIBLE_DEVICES` gives you GPU 0 and GPU 1, both empty at the start.
> Use either, or both at once for different work. One rule is hard: **your
> number and SGLang's, in the same row, must be measured on the same card in the
> same session, alternating**, with the card written in the row.
>
> There is also a page of someone else's potholes: `./TILELANG_NOTES.md`, facts
> another TileLang loop on this same H200 reproduced one by one. It is marked
> for an older version than the one you have, so treat the entries as
> hypotheses, not conclusions.
>
> **Order: evaluate the four lengths first, then decide segmentation, fusion and
> which dimensions each operator supports.** Do not fix a plan up front.
> 1. Write the prefill as authored HIR first and make `check` agree — no
>    performance number before that counts.
> 2. Evaluate **each of 128 / 256 / 512 / 1024** (`analyze` takes `--dim`): the
>    `--compute-cost` / `--roofline` / `--performance` of each, not one number.
> 3. Looking at that evidence, **decide three things yourself**: segmentation
>    (one set of cuts for all four lengths, or a fork somewhere); fusion (which
>    ops share a kernel and which do not); and **which dimensions each operator
>    supports** — a kernel may serve one length or several (128 and 256 could
>    share one, the short end padded up). Work out what one implementation fewer
>    saves and what the padding wastes. **Where you draw the line and why is the
>    answer this round asks for**, not an implementation detail.
> 4. Only then touch schedule and kernels.
>
> Leave one line per change in steps 2 and 3: what changed, and which number of
> which segment went from what to what.
>
> These four lengths are **not in the same regime**: what binds the shortest is
> not what binds the longest. `analyze --roofline` says whether each value is
> held by compute or by traffic — work out where it switches by hand, then
> measure it. **If analyze and your own arithmetic disagree, do not paper over
> it: put both numbers side by side and say which one is wrong.** Optimizing
> something that is not the binding quantity is this round's most expensive
> path.
>
> Deliverables: `model.py` (authored HIR), the scheduled HIR, the finalized TIR,
> the tilelang runtime module, `run.py`, `BENCH.md` (the table, and under it the
> three decisions and what settled them), and `WORKLOG.md`. **One chain per
> fork**: if some lengths share an implementation and others do not, each set
> must point at its own scheduled HIR and TIR and say which lengths it serves.
>
> Completion: (1) correctness by `check` — no hand-written scripts, no
> hand-picked tolerances — plus the first token identical to `transformers` at
> all four lengths; (2) all four artifacts present and corresponding; (3) the
> evaluation before the plan, with both kept; (4) **a baseline that stands up**:
> scope, the evidence that every iteration recomputed, the warm-up rule, the
> same card and session, the card number — any one of these missing and the
> table does not count; (5) **not slower than SGLang at any of the four**;
> (6) the table, five numbers per row; (7) each of the three decisions argued
> with what else was tried and what it measured.

</details>

### The second turn

> Analyse the bound types again, and try to get closer to SOL.

Sent after the table above was already measured and checked. It produced the
bound table (`reports/bound_table.json`: every kernel against its own floor at
the clock actually used, sorted into compute-, memory- and latency-bound), the
tile widths that fill whole waves of the 132 SMs, the hand warp-specialized
GEMM kernels in `runtime/kernels/gemm_ws.py`, and a faster GDN set. **Those
changes are the code in this directory; the table above is the state before
them.** Section 6 says exactly what that means.

## 3. What is here

| | |
|---|---|
| `model.py` | the authored HIR: the whole prefill. `hf_alias.py` maps the checkpoint's names, `config.json` is the model's own |
| `runtime/` | the twin (`twin.py`) and what it calls: `engine.py` (the per-length engines and their CUDA graphs), `plan.py` (the per-length tile plans), `kernels/` (the tilelang kernels: `gemm`, `gemm_ws`, `attention`, `gdn`, `small`) with the three `*_NOTES.md` that hold their variant tables |
| `scheduled/` | the scheduled HIR, **64 files**: one per (function, length), under `gemm/`, `gdn/`, `attention/`, `small/` |
| `tir/` | the finalized TIR, **64 files**, one per scheduled file, all distinct — the four implementation sets, written out |
| `run.py` | the entry point: a prompt in, the first token and the milliseconds out |
| `bench/` | the measurement itself: `alternate.py` (the driver that runs both sides on one card, alternating), `mine_worker.py`, `sglang_worker.py`, `SGLANG_NOTES.md` (how SGLang was pinned down) and `results/` — one file, every timed iteration of the run in the table, with the SM clock and the card's process list beside each call |
| `tools/` | seventeen scripts, and only what is needed to run it, check it, reproduce the numbers and regenerate the artifacts: `prepare_ckpt.py`, `hf_reference.py`, `check_hir.sh`, `check_layers.py`, `first_token.py`, `flops_ledger.py`, `bound_table.py`, `bench_table.py`, `chain_table.py`, `analyze_all.sh`, `analyze_summary.py`, `final_pipeline.sh`, `tfdeep.py` (the `tilefoundry` CLI on a 1 GiB stack: a 64-layer selection recurses deeper than CPython allows), and the four `gen_scheduled*.py` that write `scheduled/` |
| `BENCH.md` | **the run's own report**: the table, the protocol, the FLOP ledger, the correctness argument, the four-artifact chain, and the three decisions with their evidence |
| `FLOPS.md` | the FLOP and byte ledger, counted from the safetensors headers, with every formula written beside its number |
| `reports/` | one file per claim and no more: the FLOP ledger (`flops.json`), the per-kernel bound table, the segmentation matrix, the two fusion studies, the warp-specialization timings and their bitwise comparison, the `analyze` digest (`analyze/v0/SUMMARY.md`: every selection at every length with the bound it was given), the four per-kernel profiles, and the `check` transcripts |
| `negative_control/` | `model_packed_proj.py`: `model.py` with one change, the packing that reproduced a real disagreement with the reference library (`BENCH.md` section 6) |

**What is deliberately not here.** This directory keeps the artifacts, the
report and the evidence, not the run's working record. `WORKLOG.md` (46 KB, one
line per change), the 33 other probe scripts under `runtime/experiments/`, the
515 per-layer `check` transcripts (here as one `reports/check/layers_summary.json`:
how many passed and the worst case of each group), the raw per-selection
`analyze` JSON behind `SUMMARY.md`, the GEMM tile sweeps and the SOL-pass
per-case files whose results `BENCH.md` quotes inline, and every superseded run
stay in the round archive. `prompts/prompt.txt` is cut to the first 1100 tokens
— the ids of all four lengths are unchanged by that, and the rest was never
read.
`BENCH.md` cites some of them by name; the citations are left as the run wrote
them rather than edited to hide what is not shipped.

Machine paths were replaced: the published checkpoint is `$QWEN38_CKPT` in the
scripts and in the documents, and the round's own directory is `<round>` in
`bench/final_run.log`. Nothing else was edited.

## 4. Running it

```bash
export QWEN38_CKPT=/path/to/Qwen3.8-27B
python tools/prepare_ckpt.py                   # once, writes ./ckpt
python run.py --prompt "The three laws of thermodynamics are" --prompt-tokens 1024
```

The first call of a length compiles its kernels and captures its CUDA graph
(about 16 s); `--prompt-tokens` must be one of 128 / 256 / 512 / 1024, because
every kernel is compiled with the length as a static constant and `plan.for_length`
refuses anything else. That restriction is the result, not an omission: section 9
of `BENCH.md` is the measurement that says padding to the next supported length
costs 36–88% of the forward.

## 5. Correctness

```bash
export QWEN38_CKPT=/path/to/Qwen3.8-27B
python tools/hf_reference.py                      # one transformers forward per length
for T in 128 256 512 1024; do
  tools/check_hir.sh $T                           # model.py against it
  tools/check_hir.sh $T runtime/twin.py Qwen3_8_27BTwin twin_vs_hf   # the twin against it
done
python tools/check_layers.py runtime/twin.py Qwen3_8_27BTwin 128 256 512 1024 \
    --vs-hir --tag twin_vs_hir                    # the twin against the HIR, layer by layer
python tools/first_token.py
```

`tools/final_pipeline.sh` is the whole sequence as the run itself drove it.

`tilefoundry check` says all of it; no tolerance here is hand-picked. Every
bound comes from `check --help`'s own rule — `rtol = 2^-9·sqrt(n)` for `n` serial
bf16 landings, applied as `rel_l2 <= rtol` and `cosine >= 1 - rtol²/2` — plus
`nan_inf`.

| what | covers | result |
|---|---|---|
| `model.py` vs `transformers`, whole prefill path, all five outputs | that the authored HIR is the model | **PASS at 128 / 256 / 512 / 1024** |
| `model.py` vs `transformers`, per layer, teacher-forced | the same, layer by layer | **512 / 512 PASS**, worst rel_l2 0.0016 (`reports/check/hir_vs_hf_layers_summary.json`) |
| the twin vs `transformers`, whole prefill path | that the kernels compute what the HIR says, composed over 64 layers | **PASS at all four lengths**, logits rel_l2 0.0265 / 0.0214 / 0.0084 / 0.0128 |
| the twin vs the evaluator running `model.py`, per layer | the same without `transformers` in the way | **512 / 512 PASS**, worst rel_l2 0.0026 (`reports/check/twin_vs_hir_layers_summary.json`) |
| first token, twin vs `transformers` | the thing a prefill is for | **552 / 65 / 6816 / 735, identical**; max abs logit error 0.3125 / 0.3516 / 0.125 / 0.25 |

One real difference was found this way and fixed in the implementation rather
than in a bound: the twin took the two N=48 GDN projections out of the packed
`[qkv|z|b|a]` GEMM, while the reference library computes them with a split-K
sum in another order, and the GDN state's decay depends on one of them. The
negative control that isolates it is `negative_control/model_packed_proj.py`;
the fix is `gemm.ba_splitk_kernel` + `ba_reduce_kernel`, in the reference
library's own split count and order. It costs 0.39–0.68 ms per forward and it is
kept.

## 6. Where it stands

```bash
export SGLANG_PYTHON=/path/to/sglang-venv/bin/python     # sglang 0.5.21, its own torch
python bench/alternate.py --gpu 1 --lengths 128 256 512 1024 --rounds 6 --warmup 3 --iters 10
python tools/bench_table.py bench/results/<stamp>.json
```

Both sides on one card, in one session, alternating round by round; the timed
region on each side is one forward between CUDA events with a device
synchronization around it. 3 warm-up calls per side per length, then 6 rounds of
10 timed iterations, median over all 60 with the range.

| prompt | this work | SGLang 0.5.21 | vs | roofline bound | TFLOP/s | % of 989.5 | SM clock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | **19.57 ms** (19.43–19.65) | 36.84 ms | **1.88x** | 10.71 ms (memory) | 322 | 32.5% | 1500 MHz |
| 256 | **26.37 ms** (26.12–26.50) | 37.16 ms | **1.41x** | 12.73 ms (compute) | 478 | 48.3% | 1500 MHz |
| 512 | **44.02 ms** (43.72–44.25) | 48.54 ms | **1.10x** | 25.47 ms (compute) | 573 | 57.9% | 1500 MHz |
| 1024 | **82.66 ms** (82.10–83.08) | 85.41 ms | **1.03x** | 51.05 ms (compute) | 611 | 61.8% | 1485–1500 MHz |

The margin shrinks as the length grows, and that is the shape of the result
rather than noise: at 128 tokens both implementations are streaming the same
51.4 GB of weights, and what separates them is how much of the 4.8 TB/s they
reach; by 1024 both are running the same GEMMs on the same tensor cores and the
margin is 3%.

**The table is the state at which the task's criteria were met, and the code in
this directory is one optimization pass newer.** That pass — the second human
turn in section 2 — changed tile widths, added the hand warp-specialized GEMM
kernels and a faster GDN set. It is measured per kernel (`reports/bound_table.json`,
`reports/gemm_ws*.json`, `reports/gdn_sol.json`, `runtime/kernels/GEMM_WS_NOTES.md`):
every warp-specialized case is bitwise equal to the kernel it replaces, over 9
random cases, 200 repeats and a 16-layer chain; the GDN changes alter a summation
order and move the state by rel_l2 5e-6. Per-kernel sums of one forward go from
19.54 / 26.35 / 44.16 / 82.87 ms to 17.25 / 24.57 / 41.06 / 79.22 ms. **The
end-to-end re-measure of that state, and its end-to-end `check`, are not in this
directory**: another job filled all eight cards of the machine before they ran,
and the attempt is in the record as a CUDA OOM. So the honest reading of this
example is that the table's four rows were measured and checked on the state
that produced them, and the shipped kernels are faster per call than that state
without an end-to-end number of their own.

**Where the boundary is, and that it was measured.** One kernel graph serves all
four lengths, with one fork in it — the N = 5120 projections run split-K 2 where
the unsplit GEMM cannot fill the 132 SMs (both at 128, `down` only at 256, neither
at 512 and above) — and inside that graph every length has its own tile plan. The
case for it is the cross-plan matrix (`reports/segment_xplan_gpu1.json`, each
column a plan, each row a length, ms):

| T \ plan of | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|
| 128 | **19.03** | 22.09 | 25.59 | 27.28 |
| 256 | 32.79 | **25.81** | 29.14 | 31.19 |
| 512 | 57.02 | 46.93 | **43.47** | 44.40 |
| 1024 | 108.92 | 90.65 | 84.00 | **81.79** |

The best single shared plan is 512's, and it loses 34% at 128 and 13% at 256.
That is the whole case for four implementation sets, and `tir/` shows the same
answer a second way: all 64 lowered files differ.
