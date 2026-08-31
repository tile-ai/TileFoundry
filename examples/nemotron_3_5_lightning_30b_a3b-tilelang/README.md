# Nemotron-3.5-Lightning-30B-A3B on TileFoundry — one launch a decode step

Verified at **v0.0.2-dev**, 2026-08-31, against `main` at `74abc97`.
Nothing in CI re-runs it.

A 52-layer Mamba2 / attention / MoE hybrid taken end to end on TileFoundry, as
**one authored `@func` and one cooperative kernel**: token id in, next token id
out, and the whole step is a single device launch.

```
1 launch a step          against 3212 on the op-by-op path
64 of 64 greedy tokens   identical to transformers
287.4 tok/s at ctx 32    SGLang 294.8 on the same card the same day
231.6 tok/s at ctx 262080                278.2
2.37x - 2.44x off the measured bandwidth floor, flat across nine context lengths
```

**It does not beat SGLang.** It is 97.5% of it at short context and 83.2% at
262080, and the gap is accounted for in §6. What is new here is the shape, not
the speed: the authored HIR *is* the mega program, so `check` compares the thing
that actually runs.

---

## 1. Environment

Nothing here is installed by this directory; this is what it was written and
measured against.

| | |
|---|---|
| GPU | one NVIDIA H200 SXM (143 GB), **SM clock 1500 MHz** — see below |
| CUDA | 13.0, `CUDA_HOME` must be set |
| Python | 3.12 |
| `tilefoundry` | from the wheel, **not** editable — nothing here reads the source tree |
| from the environment | `torch` 2.13.0+cu130, `transformers`, `safetensors`, `tokenizers` |
| extra package | `tilelang` 0.1.13 |
| weights | the published `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, 61 GB |

**The clock is 1500 MHz, not the 1980 the card reports as its maximum.**
`nvidia-smi -lgc` needs a permission this account does not have, and the card
sits at 1500 under full load at 354 W and 43 °C, so it is neither thermal nor
power. Every number in this file was taken there, ours and SGLang's alike, and
`bench_mine.py` prints the clock on every line so a run at a different one is
visible rather than silently comparable.

### Two directories, and they are not the same thing

```bash
export NEMOTRON35_CKPT=/path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
export NEMOTRON35_PREPARED=/path/to/prepared        # what prepare_weights.py writes
python prepare_weights.py                            # once, ~2 min
```

There is deliberately no default for either: a path that exists on one machine
is not a setting. Every script also takes `--ckpt` / `--prepared`; one of the two
ways is required. `paths.py` is the only place this is decided.

## 2. How this was produced

One agent, working in a directory with nothing of ours above it.

| | |
|---|---|
| agent | Claude Code, `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint, the backend (tilelang), the deliverable filenames, and a pre-measured SGLang baseline (`SGLANG_BASELINE.md`) |
| told | everything about TileFoundry is to be asked of the `tilefoundry` command; the model itself is its own to research |
| `tilefoundry` | read-only in its venv, from the wheel built at `d73c1ee`; verified afterwards byte-for-byte unmodified (307 files) |
| human input | two turns: the prompt, then one follow-up. Both below |
| duration | 12.24 h, 1124 tool calls, no sub-agents |

### Stage 1 — the prompt, and nothing else

<details>
<summary><b>The prompt, in full</b> (translated from the Chinese it was written in)</summary>

> Take NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 to real tokens on TileFoundry,
> and make it fast.
> Weights + config: `<the checkpoint directory>` (a machine path in the original)
> Backend: tilelang. Text decode path only — the `batch=1, seq_len=1` branch.
> The `num_nextn_predict_layers` MTP layer is not part of this.
>
> **Everything about TileFoundry is to be asked of the `tilefoundry` command** —
> do not ask a person, do not go looking elsewhere. The model itself is yours to
> research.
>
> ### One shape: mega decode
>
> ```
> host:  one launch a step
>     launch(mega_decode, grid=resident, cooperative=True)
>     no CUDA graph, no PDL
>
> device: one kernel walks the whole step
>     for layer in 0 .. L-1:
>         for stage in this layer's stages:
>             do(stage)
>             barrier()        # a stage ends at a sync, not at a return
>     lm_head
> ```
>
> The prompt's tokens go through this same path one at a time. One copy of the
> weights and one of the state — no copying, no re-laying-out.
>
> ### Three constraints
>
> **One — the HIR itself has to be the mega one**, and the same shape as the
> runtime implementation. The whole decode step is *one program*; a stage's
> boundary is a mesh-level barrier, not a `@func` boundary.
>
> **Two — the kernel goes through TileLang's low-level surface.** Explicit TMA /
> mbarrier / wgmma (`T.tma_load`, `T.mbarrier_*`, `T.wgmma_gemm` and that family),
> not the `T.copy` / `T.gemm` / `T.Pipelined` sugar — where a transfer is issued,
> who waits on whom, when it lands, all of it stated out loud, because the
> cross-stage prefetch this round is about is only sayable that way.
>
> **Three — "fast" has to hold over the whole thing, not at one point.** Deliver a
> decode tok/s table from **0 to 262144** (this model's
> `max_position_embeddings`), one number per length, and every one of them has to
> stand up — an implementation that looks good at short context and falls over at
> long does not count as done.
>
> How you make the whole table stand up is **yours to decide**: whether some stage
> gets more than one split, what selects between them at run time, where the
> boundary is — measure it, decide it. But **write down what the decision rests
> on**: which ones you tried, what each measured at which lengths, why the one you
> kept is the one you kept.
>
> ### Three steps
>
> This is the way to do it with the least work:
>
> ```
> 1. write the whole decode step as authored HIR first; get a number out of
>    `analyze --performance`
> 2. explore distribution and dataflow in the HIR — placement (which mesh level
>    takes which axis, how the resident grid is spread), how stages are cut, what
>    can go in parallel, which level of storage each value lands in, where the
>    movement happens — until that number stops dropping
> 3. only then touch the kernel
> ```
>
> Leave one line per change in step 2: what changed, and the number from-to.
> Before step 3, report that last number **together with** the roofline floor.
> **`analyze` prices the bytes moved** — following it on a stage that is not
> bandwidth-bound is wasted work. If the two numbers do not meet, say so rather
> than forcing them together.
>
> ### Deliverables
>
> ```
> model.py          TileFoundry Module: the HIR of that mega decode step
> runtime_model.py  its twin; the kernel is called from here
> run.py            entry: python run.py --prompt "..." --max-new-tokens 512
>                   prints the continuation and a tok/s number;
>                   also supports --seed and --greedy
> ```
>
> ### Done when
>
> In order — a later row's numbers do not count until the earlier one passes.
>
> | | criterion | what counts as passing |
> |---|---|---|
> | 1 | correct | `check` says so — **no hand-written comparison, no self-chosen tolerance**. It covers the whole model; the greedy tokens match `transformers` **id for id, out to dozens of steps** |
> | 2 | both sides the same shape | in the twin pair `check` compares, the semantic side's decode step is **one** mega program; you can say where its stage boundaries and barriers are |
> | 3 | one launch | launches per step **= 1**, reported **together with** the same number for the op-by-op path |
> | 4 | `analyze` pushed to the floor | the record from step 2 is there; the final number is reported together with the roofline floor |
> | 5 | that table | steady-state tok/s, **one number per length from 0 to 262144**, three rows side by side: **yours**, **SGLang's**, **the roofline floor** (floor = measured bytes per token at that length ÷ HBM bandwidth). Measure SGLang yourself with `~/sglang-bench` (its `.venv` is ready) on the **same card and the same weights** — do not quote numbers from anywhere else. Pin the card, warm the clocks first |
> | 6 | the choices have grounds | for any part of the table, you can say why the current version is the one there: which ones you tried, what each measured. **Reporting only the final one does not count** |
>
> You may also stop, on any one of these:
>
> - you judge that everything this command surface can be asked has been asked,
>   and going further is guessing
> - three times stuck on the same kind of thing
> - the loop is turning smoothly and what is left is repetition

</details>

**Where it got to on the prompt alone.** The mega shape, both generators, a
five-rung HIR placement ladder (`gen_model.py --variant 0..4`, each rung
re-analyzed), and sixteen kernel changes each carrying its own number — the step
went from 9.130 ms to 3.556 ms at short context and 18.153 to 7.723 at 262080.
Three of those sixteen were tried, measured slower, and backed out. That is this
table:

| context | 0 | 32 | 1024 | 4096 | 16384 | 32768 | 65536 | 131072 | 262080 |
|---|---|---|---|---|---|---|---|---|---|
| tok/s | 278.7 | 277.5 | 271.9 | 257.8 | 255.0 | 239.2 | 212.7 | 174.7 | **128.7** |
| off the floor | 2.46x | 2.47x | 2.52x | 2.65x | 2.65x | 2.79x | 3.05x | 3.52x | **4.33x** |

Correct everywhere and one launch everywhere, but **the last row degrades with
length**: 2.46x at the short end, 4.33x at 262080. One attention placement was
being asked to serve both ends of an 8000x range, and it could not.

### Stage 2 — one follow-up

> Is the time mainly going to attention? Then pull that layer out as a module of
> its own and dispatch on length — split by query head when the sequence is short,
> by context when it is long — and put that back into the step. Go write that
> kernel. Also: why is 256K so much worse than the rest, and why is this not on
> wgmma?

Two things in the shipped result come from this and not from the prompt: the
**length dispatch** (`attention.py`, §5) and the move of the attention tile
**onto tensor cores**. `attention.py` is what it did with the first: the layer
pulled out on its own with three placements (`by_head`, `by_context`, `by_both`)
analyzed apart, and the crossover **read off `analyze` at ctx 2049** rather than
tuned — a hand estimate had said 4096, twice too far, because it forgot that
`by_context` also pads a tail block. The shipped dispatch turns over at 2048.

The result of that second stage is the table in §6: **2.37x–2.44x, flat across
all nine lengths**, and 262080 from 128.7 to 231.6 tok/s — **1.80x**.

## 3. What is here

| | |
|---|---|
| `model.py` | the authored HIR: the whole decode step as one `@func decode_step`, 474 parameters |
| `runtime_model.py` | its twin — same `decode_step`, one of two implementations |
| `mega_kernel.py` | that one launch, in TileLang |
| `gen_model.py` `gen_kernel.py` | **the generators. Change these, not what they emit** |
| `attention.py` | the attention layer alone, as a placement ladder (§5) |
| `run.py` `generation.py` | continuation and timing |
| `check_all.py` | `tilefoundry check` spread over all 59 outputs, bounds derived per output |
| `compare_transformers.py` | token by token against `transformers` |
| `bench_mine.py` `bench_sglang.py` `make_table.py` | the two sides of the table |
| `tools/` | launch count, mega-vs-ops, the long-context timing path |
| `kbench/` | the measurement behind each "why it is written this way" |
| `ISSUES.md` `repro/` | what TileFoundry and TileLang got in the way with, each with a minimal repro |

**Why two generators.** TileLang rewrites every `for` in a `@T.prim_func` into a
device loop (`range` is overridden to `T.serial`) and does not bind a nested
`def`; HIR's `for` is a runtime loop over one weight tensor. So repetition that
has to happen at trace time — a prefetch prologue, a fixed number of accumulator
registers, one body per stage kind, and 52 layers that share neither weights nor
shape — has to be *written out*. `model.py` (276 KB) and `mega_kernel.py`
(100 KB) are that writing-out, and are not meant to be edited by hand.

## 4. Running it

```bash
python run.py --prompt "The three laws of thermodynamics are" --max-new-tokens 512 --greedy
```

`PYTHONPATH=$PWD` is required for anything that goes through
`tilefoundry check`: it runs the target in a subprocess that would not otherwise
find `mega_kernel`.

## 5. Correctness, and exactly what each check covers

```bash
PYTHONPATH=$PWD python check_all.py --ctx-full 0 --ctx-tail 128
python compare_transformers.py --impl mega --steps 64 \
       --prompt "The three laws of thermodynamics are"
PYTHONPATH=$PWD python tools/ctxcmp.py 300
python tools/launches.py
```

| what | covers | result at `74abc97` |
|---|---|---|
| `check_all.py` — twin against the authored HIR, 59 outputs, each with a bound argued from how many bf16 landings are on the path to it | the step's arithmetic, **short-context arm only** | **59 of 59 pass** |
| `compare_transformers.py` | both arms, end to end | **64 of 64 tokens identical** |
| `tools/ctxcmp.py` — mega against op-by-op across the `ABLK` boundary, 300 steps | the long-context arm | 9 of 11 sampled lengths agree; the 2 that differ are the two where the reference's own top-2 gap is 0.0312, **one bf16 ulp** |
| `tools/launches.py` | the shape claim | **1** launch a step, against **3212** |

**`check` only runs at `ctx_full = 0`, and that is a limit of the tool, not a
choice.** Splitting a long reduction over a worker axis — the only placement with
enough parallel units for long-context attention — writes as `b0 = t + kv.w*BLK`,
and the evaluator refuses it:

```
Local on a Split axis is not modelled: evaluation runs one mesh participant
(docs/spec/evaluator.md section 6)
```

At `ctx_full = 0` the tile loop runs zero times and the path is never taken.
So the long-context arm's numerical correctness is **not** established by
`check`; it rests on the two rows below it. This is being fixed upstream; when it
is, `check_all.py --ctx-full 262016` becomes the test that says so.

## 6. Where it stands

```bash
CUDA_VISIBLE_DEVICES=0 python bench_mine.py --out reports/mine.json
CUDA_VISIBLE_DEVICES=0 python bench_sglang.py \
    '{"mem_fraction_static":0.85,"disable_radix_cache":true}' \
    '32,1024,4096,16384,32768,65536,131072,262080' reports/sglang.json
python make_table.py --mine reports/mine.json --sglang reports/sglang.json
```

Three rows, one H200, one set of weights, one day. SGLang is measured here too,
not quoted — with everything it turns on by default (`fa3`, `flashinfer_cutlass`
MoE, CUDA graphs, overlap schedule) and **no speculative decoding on either
side**, since this implementation does not do MTP. The method is in
`SGLANG_BASELINE.md`.

| context | 0 | 32 | 1024 | 4096 | 16384 | 32768 | 65536 | 131072 | 262080 |
|---|---|---|---|---|---|---|---|---|---|
| **this, tok/s** | **288.8** | **287.4** | **282.6** | **283.1** | **277.4** | **273.5** | **266.0** | **254.1** | **231.6** |
| SGLang, tok/s | — | 294.8 | 293.6 | 292.3 | 291.4 | 288.7 | 283.2 | 288.2 | 278.2 |
| roofline floor, tok/s | 686 | 686 | 685 | 684 | 676 | 667 | 649 | 615 | 558 |
| bytes a token, GB | 6.997 | 6.998 | 7.004 | 7.023 | 7.098 | 7.199 | 7.400 | 7.803 | 8.608 |
| **off the measured floor** | **2.37x** | **2.39x** | **2.42x** | **2.41x** | **2.44x** | **2.44x** | **2.44x** | **2.42x** | **2.41x** |

The numbers are the run's own. Six of the nine lengths were re-measured at
`74abc97` while preparing this directory and agree to **0.1%** (288.92 / 287.63 /
282.65 / 283.14 / 277.69 / 273.69 against the row above); the remaining three
were not re-run.

**The last row is the point of the table.** 2.37x to 2.44x, flat across nine
lengths. Before the attention went onto tensor cores it ran 2.46x to **4.33x** —
degrading with length. Flat means no length is especially bad any more, and that
the remaining gap is *one* thing rather than several.

That one thing was measured rather than guessed at: about half of the 2.4x is
issue density, not bandwidth — a clock sweep buys +16% for +32% of clock, which
is what a half-and-half split looks like. The gemv that dominates a step computes
each output row as a 132-way split dot product, 21 rows by 2688 long per CTA, and
the two ways at it — numbering tiles globally, and letting the prefetch run
across a stage boundary without draining — are estimated but not built. The run
stopped there because its subject was attention.

**The 262080 point is measured differently and here is why.** The other eight
walk a real decode out to that length and then time 256 steps. 262080 never
finished: from 131072 those 131k steps should be nine minutes at the 3.935 ms
that same script reports at 131072, and it ran over two hours with the machine
otherwise idle. That was never explained, and is not claimed to be. The kernel
side is clean — a counter dialled along the same leg over seven lengths rises
smoothly to 4.32 ms with no bad length — so the problem is in that script's host
loop, not in what ships. That point uses `tools/ctxtime.py`, which allocates the
cache at full capacity and dials `caches["step"]`: same bytes read, same
iterations, different values in the cache. The two methods agree to **0.4%** at
all eight lengths where both can run.

## 7. What got in the way

`ISSUES.md` — 7 in TileFoundry, 8 in TileLang, each with a minimal repro under
`repro/` or `kbench/`. Two are worth reading before you write anything similar:

* **TF-1**, above: the evaluator models one mesh participant, so the placement
  long-context attention needs cannot be `check`ed. Being fixed.
* **TL-3**: a gemm benchmark whose operands do not change gets hoisted out of the
  loop and measures something 6x faster than the real thing. Both the wrong and
  the right harness are in `kbench/`.

`TF-3` (a diagnostic that named neither file nor line) **was fixed** in `#141`;
the error quoted in §5 is what it says now.
