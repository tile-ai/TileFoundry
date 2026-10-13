# Qwen3.8-27B prefill on TileFoundry — faster than SGLang at every length

Verified at **v0.0.2-dev**, 2026-10-13, against `main` at `f8fd71ea`: `run.py`
and the whole table re-measured from this directory in a fresh virtualenv
(section 6). Nothing in CI re-runs it.

A 64-layer gated-delta-net / attention hybrid, prefill only: token ids in,
first token out. Batch 1, no history, 1024 / 4096 / 8192 tokens, bf16, no
quantization, no vision, no decode.

```
80.29 / 309.61 / 631.19 ms     SGLang 0.5.21: 89.06 / 317.11 / 638.25
-9.8% / -2.4% / -1.1%          on the same card, the same weights, the same scope
634.6 / 666.2 / 664.1 TFLOP/s  64% / 67% / 67% of the H200's 989
first token identical to transformers at all three lengths
```

**This one is compute bound**, which inverts the usual problem: at 8192 the
arithmetic intensity is about 36x the H200's roofline ridge, attention scores
are 3.1% of the FLOP, and what is left is almost pure GEMM. There is no traffic
to remove — the floor is `FLOP / peak`, and the whole result is how close to it
the GEMMs get.

The full argument for every number is in [`BENCH.md`](BENCH.md), which is the
run's own report kept whole. This file is the front door.

---

## 1. Environment

Nothing here is installed by this directory; this is what it was written and
measured against.

| | |
|---|---|
| GPU | one NVIDIA H200 SXM (143 GB), **SM clock 1500 MHz** — see below |
| CUDA | 13.2, `CUDA_HOME` must be set |
| Python | 3.12 |
| `tilefoundry` | `0.0.2+32.gf8fd71ea`, from the wheel, **not** editable — nothing here reads the source tree |
| from the environment | `torch` 2.14.1+cu130, `transformers` 5.14.1, `safetensors`, `tokenizers` |
| extra package | `tilelang` 0.1.15 |
| weights | the published `Qwen3.8-27B`, bf16 |

**The clock is 1500 MHz, not the 1980 the card reports as its maximum.**
Locking it needs a permission this account does not have; the card sits at
1500 under load. `tools/bench.py` reads the SM clock after every timed replay
and puts it in the row, so a run at another clock is visible rather than
silently comparable. The 8192 row's clock drifts to 1365–1500 under sustained
load, which is why its range is wider than the others'.

### Two directories, and they are not the same thing

```bash
export QWEN38_CKPT=/path/to/Qwen3.8-27B        # the published checkpoint
export QWEN38_REFDATA=/path/to/refdata         # what tools/hf_reference.py writes
```

There is deliberately no default for either: a path that exists on one machine
is not a setting. Every script also takes `--ckpt` / `--refdata`; one of the two
ways is required. `paths.py` is the only place this is decided.

`QWEN38_REFDATA` is needed **only by the correctness checks** (section 4). It is
one `transformers` forward pass recorded tensor by tensor, about 20 GB, which is
why it is not shipped:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/hf_reference.py \
    --lens 1024,4096,8192 --layers 0,3 --out "$QWEN38_REFDATA" \
    --ckpt-out "$QWEN38_REFDATA/ckpt"
```

## 2. How this was produced

One agent, working in a directory with nothing of ours above it.

| | |
|---|---|
| agent | Claude Code, `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint, the backend (tilelang), the deliverable filenames, a pre-measured SGLang baseline (`SGLANG_BASELINE.md`), its `flops.py` and `bench_prefill.py`, and a page of TileLang gotchas from another loop on the same card |
| told | everything about TileFoundry is to be asked of the `tilefoundry` command; the model itself is its own to research; **do not measure SGLang — read the table** |
| `tilefoundry` | read-only in its venv, from the wheel built at `f8fd71ea`; verified afterwards byte for byte unmodified (332 files) |
| human input | two turns: the prompt, then one sentence about which GPU to use. Both below |
| duration | 14.88 h, 1434 tool calls, no sub-agents, 5 context compactions |

**The baseline was handed over, not re-measured.** Earlier examples let the run
measure both sides; this one gives it the opponent's table as a fact and asks
only for its own half. What the run still had to do with that table is decide
*which column it is comparable to* — the baseline carries two, forward-only and
end-to-end, each with its own flaw written down, and a 9.1% contradiction
between them at 8192 that is stated and deliberately not explained. Picking the
line, and defending the pick, was part of the task; the answer is `BENCH.md`
section 3.

### Stage 1 — the prompt, and nothing else

<details>
<summary><b>The prompt, in full</b> (translated from the Chinese it was written in)</summary>

> Get Qwen3.8-27B's **prefill** right on TileFoundry, and make it fast.
> Weights + config: `<checkpoint>`
> Backend: tilelang.
>
> Text prefill main path only: **batch = 1, history = 0**, prompt lengths
> **1024 / 4096 / 8192**. No decode (stop after prefill emits the first token);
> no vision tower; no quantization.
>
> Everything about TileFoundry is to be asked of the `tilefoundry` command — do
> not ask a person, do not go looking elsewhere. The model itself is yours to
> research.
>
> **The opponent is already measured, in `./SGLANG_BASELINE.md`.** Same card,
> same weights, SGLang 0.5.21's prefill latency table. **You do not measure
> SGLang** — this round you implement only your own half, then measure yourself
> by the protocol in that file and put the rows side by side.
>
> Three things are in that file, all three for you to use:
>
> - **Two columns**: forward only and end to end, each with its own protocol and
>   its own flaw, both written down. Also a fact that the two columns disagree
>   by 9.1% at 8192, left unexplained — that is yours to judge.
> - **The protocol**: how a measurement counts. Copy it, it is not a suggestion
>   — **both sides count forward only**, the serving path's fixed costs count on
>   neither; 2 untimed warmups + 7 timed, median, read the SM clock each time,
>   report the range beside the median. The warmup especially cannot be skipped
>   — skip it and 8192 comes out 20% slow.
> - **An account to re-check yourself**: `flops.py` is in the directory, it
>   counts FLOP from the safetensors headers rather than trusting the model
>   card. Your TFLOP/s and your compute floor both come out of it.
>
> `bench_prefill.py` is here too; it is the *definition* of that protocol.
> Follow it when you measure yourself.
>
> **A page of someone else's scars: `./TILELANG_NOTES.md`.** Facts another
> TileLang loop on this same H200 reproduced one by one — WGMMA descriptor
> R2UR, the barrier convention for hand-written warp specialization, the shared
> memory allocator's temper, several ways to write it that simply fail. **With
> reproducers and build numbers.** Two warnings: the entries are marked
> `[0.1.11]` / `[0.1.12]` and **you have 0.1.15** — they are hypotheses, not
> conclusions; try one when you hit its symptom and write down what happened.
> It is only about TileLang; **TileFoundry's command surface is still something
> you ask `tilefoundry` about.**
>
> **Order: evaluate the three length segments first, then decide how to segment
> and how to fuse.** Do not settle on a plan up front. This round explicitly
> wants evidence before the cut:
>
> 1. Write prefill as authored HIR first and make `check` agree — any
>    performance number before this step does not count.
> 2. Evaluate **each of 1024 / 4096 / 8192** (`analyze` takes `--dim` to bind
>    the length). What is wanted is each segment's own `--compute-cost` /
>    `--roofline` / `--performance`, not one number.
> 3. Looking at those three, **decide the segmentation and the fusion plan
>    yourself**: do the three lengths share one cut and one fusion, or does it
>    fork somewhere into two or three; inside each segment, which ops belong in
>    one kernel and which do not. **Where you draw the boundary and why is the
>    answer this round wants**, not an implementation detail.
> 4. Only then touch schedules and kernels.
>
> For steps 2 and 3, leave one line per change: what changed, and which number
> at which length went from what to what.
>
> This shape is **compute bound**, the reverse of what is usual here: at 8192
> the arithmetic intensity is about 36x the H200's ridge (the account is in
> `SGLANG_BASELINE.md`, re-check it). `analyze --roofline` tells you whether each
> value is held by compute or by memory — **if it says memory and your hand
> count says compute, do not paper over it: put the two numbers side by side and
> say which one is wrong.** Optimizing along a quantity that is not the binding
> one is this round's most expensive way to go.
>
> **Deliverables.** One chain, every link kept and pointed at:
>
> ```
> model.py           authored HIR -- prefill's reference description; what `check` checks
> scheduled HIR      the one after schedule and instruction selection
> finalized TIR      what `schedule finalize` produced, verified
> runtime module     the tilelang implementation; kernels are called from here
> run.py             entry: python run.py --prompt "..." [--prompt-tokens 8192]
>                    prints prefill's first token, and prefill's milliseconds / tok/s
> BENCH.md           the table, and under it your segmentation and fusion plan
>                    with the evidence that settled it
> ```
>
> The four program artifacts must correspond: you can say which TIR came from
> which scheduled HIR, and which scheduled HIR from which part of `model.py`.
>
> **Done when**, in order — a number from a later row does not count until the
> earlier ones hold:
>
> | | criterion | what counts as passing |
> |---|---|---|
> | 1 | correctness holds | `check` says so — **no scripts of your own, no tolerances of your own**. Cover the whole prefill path. And one more: the first token out of prefill is identical to `transformers` at all three lengths, with the max absolute logit error reported |
> | 2 | four artifacts present | authored HIR / scheduled HIR / finalized TIR / tilelang runtime module all there and corresponding. The semantic side is real HIR, not a shell |
> | 3 | evaluation came first | the three per-length `analyze` reports from step 2 are kept, and so is step 3's running "what changed, from what to what". **Evidence before plan**; the other order does not count |
> | 4 | not slower than SGLang | **each** of the three lengths. `./SGLANG_BASELINE.md` gives you **facts**: two columns from two instruments, each protocol, each flaw. **Which column, which number is the line comparable to yours is yours to judge** — write the reason down, then nail it. No picking a different instrument per length. Judging it wrong is worse than falling short. If you fall short, say by how much and where it is stuck |
> | 5 | that table | three rows, five numbers side by side on each: **your ms**, **SGLang's ms**, **the compute floor**, **your TFLOP/s**, **its percentage of 989**. Copy the protocol: same card, 2 warmups + 7 timed median, range and SM clock on the table. Be able to say what your timer covers, from where to where (no weight loading, no compilation, no tokenization) |
> | 6 | the segmentation is argued | every boundary in the final plan can be explained: which alternatives were tried, and what number each gave at which lengths. **Reporting only the final one does not count** |
>
> You may also stop short, on any of these: you judge that everything this
> command surface can answer has been asked and going further is guessing;
> three consecutive blocks on the same kind of thing; the loop has become
> routine and what is left is repetition.

</details>

### Stage 2 — one sentence

> You can pick another clock-locked GPU, you do not have to wait for this one.

Sent mid-run, because another user's process had taken the card. The run moved
development to GPUs 0 and 1 and kept the final table on the pinned card — the
same card `SGLANG_BASELINE.md` was measured on, which is the only thing that
makes the two columns comparable.

## 3. What is here

| | |
|---|---|
| `model.py` | the authored HIR: the whole prefill, `hf_alias()` maps the checkpoint's names |
| `runtime_model.py` | its twin — the same functions, implemented by calling kernels |
| `kernels/` | the TileLang kernels: `gemm`, `attention`, `gdn`, `elementwise`, `pack` |
| `scheduled/` | the scheduled HIR, one Module per (function, length) |
| `tir/` | the finalized TIR. **35 files stand for 75** — see `tir/INDEX.md`, where the de-duplication is itself the segmentation result |
| `run.py` | the entry point: a prompt in, the first token and the milliseconds out |
| `tools/` | eleven scripts, and only what is needed to run it, check it, reproduce the table and regenerate the TIR: `bench.py`, `check_hir.sh`, `check_twin.sh`, `e2e_twin.py`, `hf_reference.py`, `finalize_all.sh` (+ `finalize_deep.py`, `scheduled_index.py`), `analyze_all.sh`, `segment_study.sh`, `prompt.py` |
| `BENCH.md` | **the run's own report**: the table, the protocol, the correctness argument, the correspondence of the four artifacts, and the segmentation with its evidence |
| `SGLANG_BASELINE.md` | the opponent, measured beforehand and handed over: two columns, each protocol, each flaw |
| `bench_prefill.py` `flops.py` | the SGLang side of the protocol, and the FLOP count both sides use |
| `reports/` | JSON only: the eight benchmark runs every number in `BENCH.md` section 1 rests on, the two segmentation studies behind section 6, and the SGLang baseline's raw numbers. Everything in `tools/` writes here |

**What is deliberately not here.** This directory keeps the artifacts and the
report, not the run's working record. The per-change log (`evidence/STEP3.md`,
`STEP4.md`), the per-function `analyze` reports, the `check` transcripts, the
nsys traces, the placement ladders and the per-kernel probes stay in the round
archive. `BENCH.md` and a few source comments still cite them by name — the
comments are left as the run wrote them rather than edited to hide the
citation.

## 4. Running it

```bash
export QWEN38_CKPT=/path/to/Qwen3.8-27B
python run.py --prompt "The three laws of thermodynamics are" --prompt-tokens 8192
```

The first run compiles the kernels. Prompts are padded up to a multiple of 128
(the kernels' row tile); every layer is causal, so the padding does not change
the rows before it, and the logits are read at the last real row.

## 5. Correctness, and exactly what each check covers

```bash
export QWEN38_REFDATA=/path/to/refdata          # section 1
CUDA_VISIBLE_DEVICES=0 tools/check_hir.sh 1024 "0 3"
CUDA_VISIBLE_DEVICES=0 tools/check_twin.sh 1024 "0 3"
CUDA_VISIBLE_DEVICES=0 python tools/e2e_twin.py --lens 1024,4096,8192 --graph
```

| what | covers | result |
|---|---|---|
| `check_hir.sh` — the authored HIR against the published model's own activations, 18 functions of a linear-attention layer and a full-attention layer, plus `embed` and `final` | that `model.py` is the model | **18 / 18 PASS at each of the three lengths** |
| `check_twin.sh` — the twin against `model.py` with no `--expected`: the reference is the evaluator running the authored HIR on the same inputs | that the kernels compute what the HIR says | **54 / 54 PASS** (18 functions x 3 lengths) |
| `e2e_twin.py` — the whole prefill against `transformers` | both, composed over 64 layers | **first token identical at 1024 / 4096 / 8192**; max abs logit error 0.25 / 0.38 / 0.85 |

**Every tolerance is derived, not tuned, and the one that was not is marked as
such.** `tools/check_twin.sh`'s header argues each bound from where the twin
rounds or sums in another order — one bf16 landing is one step, 2^-7 of the
value, so `rtol = 2^-7`; residual outputs get `atol = 2^-7 max|acc|`; attention
gets `atol = 2^-7 max|v|` because the twin lands the softmax weights in bf16
before the `1/l` division and `model.py` after. The exception is `delta_rule`,
whose first bound was too tight and was widened *after* it failed — so its
margin is stated rather than assumed: measured max error 0.031 against the bound
0.073 at all three lengths. `BENCH.md` section 6 tells that story in full,
including the eight functions whose first failure was the derivation's fault and
not the kernels'.

**What the end-to-end number does and does not say.** Changing nothing but
attention's key-block size — same arithmetic, other blocking — moved the 8192
logit error between 0.85 and 1.50. So it measures how rounding differences grow
through 64 layers, not the quality of one kernel. The first token's lead over
the second is 6 / 11 / 11 logits, at least 13x the largest error seen.

## 6. Where it stands

```bash
CUDA_VISIBLE_DEVICES=0 python tools/bench.py --out reports/mine.json
```

Same card, same weights, same day, same scope on both sides: one forward pass,
from token ids on the device to the first token on the device. No scheduler, no
sampling server, no detokenizer, no IPC — on either side. 2 untimed warmups
then 7 timed, median reported with the range; every repeat uses a different
prompt. SGLang runs with everything it turns on by default and no speculative
decoding; the method is in `SGLANG_BASELINE.md`.

| prompt | this work | SGLang 0.5.21 | vs | compute floor | TFLOP/s | % of 989 | SM clock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | **80.29 ms** (80.02–82.61) | 89.06 ms | **−9.8%** | 51.5 ms | 634.6 | 64.2% | 1500 MHz |
| 4096 | **309.61 ms** (306.90–315.89) | 317.11 ms | **−2.4%** | 208.6 ms | 666.2 | 67.4% | 1500 MHz |
| 8192 | **631.19 ms** (629.11–670.55) | 638.25 ms | **−1.1%** | 423.8 ms | 664.1 | 67.1% | 1440 MHz |

Which SGLang column this is, and why that one, is `BENCH.md` section 3 — the
choice was the run's and it is argued there. The margin at 8192 is 7.1 ms, and
the last seven builds on this card put the median between 629.6 and 632.2 ms,
so the ordering is stable but the 8192 row is the one to re-measure first if
anything changes.

**The three rows were re-measured from this directory** while it was being
prepared — a fresh virtualenv, the paths resolved through `paths.py`, nothing
of the run's working tree in reach (`reports/bench-reverify-2026-10-13.json`):

| | 1024 | 4096 | 8192 |
|---|---:|---:|---:|
| the table above | 80.29 ms | 309.61 ms | 631.19 ms |
| re-measured, 2026-10-13 | 80.70 ms | 310.09 ms | 630.63 ms |
| | +0.5% | +0.2% | −0.1% |

Still ahead of SGLang at all three: −9.4% / −2.2% / −1.2%.

**Where the boundary is, and that it was measured.** The segmentation forks in
exactly one place: the GEMM decomposition. Everything else — every fusion — is
one configuration at all three lengths. The cut is between 1024 and 4096, and
the study behind it swept seven lengths with the decomposition forced to each
configuration (`reports/segment2/`, `tools/segment_study.sh`):

| n | GROUP 8, split tail | GROUP 8, no split | GROUP 16, split | GROUP 16, no split | the shipped rule |
|---:|---:|---:|---:|---:|---:|
| 1024 | 79.97 | 94.54 | = g8 | = g8 | **79.89** |
| 2048 | 155.50 | 162.79 | 154.61 | 162.32 | **154.24** |
| 4096 | 306.46 | 306.35 | 305.22 | 304.78 | **304.93** |
| 8192 | 619.67 | 620.09 | 619.89 | 621.59 | **620.85** |

At 1024 the wrong choice costs **18%**; by 4096 all four are within 0.6% and
the shipped rule is within 0.2% of the best forced configuration. That is the
whole case for cutting there and nowhere else — and `tir/INDEX.md` shows the
same boundary a second way, in the lowered code: the GEMMs are the only
functions whose TIR differs between 1024 and 4096, and 4096 and 8192 lower
identically.

A first run of that study is discarded and said so: another user's process came
onto the card during one configuration and moved those cells by 15–28%. The
per-run process list in the JSON is what caught it.
