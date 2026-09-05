# Why the handwritten kernel is slower than the TileLang one

Measured on one H200, one session, `torch 2.14.0+cu130`. The TileLang numbers are
re-measured here rather than quoted: the figures in `README.md:301` were taken on
another day and are **16–17% below** what the same kernel does today, so using
them as the bar would let a 17%-slower implementation look level.

| ctx | `README.md:301` records | measured today |
|---|---|---|
| 32 | 287.4 tok/s | **335.87 tok/s** |
| 262080 | 231.6 tok/s | **268.19 tok/s** |

## 1. Where it stands

| | ctx 32 | ms/token | |
|---|---|---|---|
| TileLang, one cooperative launch | **335.9 tok/s** | 2.977 | faster |
| handwritten, one cooperative launch | **153.6 tok/s** | 6.510 | **2.19x slower** |

Slower, throughout this document: the handwritten kernel takes 2.19 times as
long to produce a token as the TileLang one does. The goal was a handwritten
kernel **no slower** than TileLang, so the goal is not met.

End to end through `bench_mine.py`, which is the number to quote. Timing the
launch alone in a tight loop gives 5.495 ms; the difference is the Python either
implementation pays per step for `prepare_inputs_for_generation` and
`append_cache`, and both pay it.

It is correct: against the TileLang kernel the whole step lands at 4.513e-2 on
the 5.018e-2 envelope `check_all.py` derives, with the same argmax at every step.
Greedy token identity is a knife edge that all three implementations fall off on
some prompt and none falls off consistently -- the table is in [section
7](#7-token-identity-is-a-knife-edge-not-a-gate). Speed is the whole of what is
wrong here.

## 2. Which layers

Inside the launch, by timing prefixes of the layer walk:

| layer kind | count | in the fused launch | as separate launches | ratio |
|---|---|---|---|---|
| Mamba-2 | 23 | 63.9 us | 49.8 us | 1.28x |
| MoE | 23 | 123.1 us | 114.2 us | 1.08x |
| **attention** | **6** | **260.7 us** | **57.1 us** | **4.6x** |
| closing norm + head | 1 | ~240 us | — | — |

Weighted: MoE is 2.83 ms of the 5.49, attention 1.56 ms, Mamba-2 1.47 ms.

## 3. What each gap is

**The grid barrier costs about 2–3 us.** Mamba-2 pays 14.1 us over five barriers
and MoE 8.9 us over five, which is the whole difference between fused and
separate for those two. That is the price of being resident and it is a small
one.

**Attention does not fit that account.** Four barriers cannot be 203 us. At
ctx 32 the scan is split over the context — `nb_tail * HKV` is 2 — so two CTAs
do the scan and 130 wait at the barrier behind them. The split that long context
needs is the wrong one at short context, which is the crossover
`attention.py` dispatches on and which this kernel does not implement: it has one
placement where the authored program has two.

**MoE is the largest absolute cost and it is bandwidth, not structure.** Its four
expert projections move 160 MB a layer and take 74 us of the 114, which is where
the TileLang kernel's whole-step budget implies they should be. The remaining
40 us is the gap between six launches and one.

**The step as a whole moves 6.997 GB a token** (`README.md:302`). TileLang gets
2.35 TB/s of that. This kernel gets 1.27 TB/s.

## 4. What the two implementations do differently

The TileLang kernel stages every projection's weight slice through shared memory
three tiles ahead. This one does that for the Mamba-2 projections and the head,
and reads the expert weights straight from global memory, because measured
separately the choice inverts with how many tiles a CTA gets:

| | staged | direct |
|---|---|---|
| Mamba-2 layer (about 78 rows a CTA) | **49.6 us** | 85.7 us |
| MoE layer (under 2 tiles a CTA) | 320 us | **114 us** |

In one resident kernel that trade-off is not free to make per stage: the launch
has a single shared-memory request, and asking for the staged ring's 131 KB
holds the grid at one CTA per SM for every stage, including the ones that would
rather have four. Dropping the staged path from the two projections that did not
need it took the step from 6.77 ms to 5.49 ms — 125 to 182 tok/s — without
changing a single arithmetic result.

## 5. Hypotheses that were wrong

Recorded because the wrong ones outnumbered the right one four to two, and each
was plausible enough to act on:

| guessed | measured |
|---|---|
| twelve expert launches are the cost | fusing them to two: **13% faster** |
| shared-memory staging starves occupancy | removing it from MoE: **no change** |
| the accumulator dependency chain stalls the loads | eight accumulators: **3% faster** |
| the router is fine, it reads 688 KB | it was **236 of the 319 us** — one thread per row, uncoalesced, one CTA |
| aliasing `__restrict__` on the in-place residual explains the drift | it does not; the drift is the expert accumulator's summation order |
| allocating 46 state tensors a step costs the missing millisecond | pooling them: **no change** |

Only `nsys` and the layer-prefix timings moved this forward; every hypothesis
formed by reading the code was wrong.

## 6. What is fixable and what is not

- **Attention's placement (fixable, ~1.3 ms).** Splitting by head at short
  context, the way the authored program does, would put the other 130 CTAs to
  work. The crossover `attention.py` already names is the number to dispatch on.
- **The shared-memory ceiling (fixable, unquantified).** A staged path that fits
  in a smaller ring, or a head that stages less, would let more CTAs per SM and
  lift the expert projections nearer their 74 us floor.
- **The barrier count (not worth it).** 5 barriers a layer at 2–3 us is about
  0.7 ms of the 5.49. Merging stages to save some of it costs the clarity of
  having one barrier per reshard the authored program states.

## 7. Token identity is a knife edge, not a gate

An earlier reading of this said the fused path had a fault of its own, because it
diverged from `transformers` where the stage-per-launch path did not. That
compared two different builds. Measured together — 48 steps, greedy,
teacher-forced, three prompts:

| prompt | `mega` (TileLang) | `cuda` (fused) | `cuda-stages` |
|---|---|---|---|
| `The capital of France is` | 48/48 | step 35 | step 35 |
| `In 1969 the first humans` | step 3 | step 3 | step 3 |
| `A prime number is` | step 42 | 48/48 | step 42 |

Every implementation loses it on some prompt. On the middle one all three
diverge at the same step and pick the same token as each other — a tie the
reference resolves one way and all three of these resolve the other. Nothing here
separates the handwritten kernels from the shipped one, and `README.md`'s "64 of
64" is the first prompt with the TileLang kernel rather than a property any of
them has in general.

The two CUDA paths do differ from each other by a hair, and it is worth naming
because it is the only genuine arithmetic difference between them. The six routed
experts sum into an f32 accumulator: six concurrent CTAs in the staged path, one
CTA six times in the fused path. On one layer, every other intermediate is
bit-identical and the accumulator differs by 5.8e-8. Across the stack:

| after layer | rel_l2 (fused against staged) |
|---|---|
| 1 – 9 | **0** (bit-identical) |
| 10 | 4.1e-4 |
| 12 – 33 | 5–7e-4 (flat) |
| 40 | 7.6e-3 |

Nine layers bit-identical, then parts in 1e4: that is 5.8e-8 reaching a bf16
rounding boundary, not a fault. Neither order is more correct, and the fused
one is the order the op-by-op reference uses.

Two other explanations were tested and eliminated. The fused launch is
**deterministic** — three runs bit-identical — so nothing races. And filling
every scratch buffer with NaN before a step changes **nothing**, so no stage
reads a buffer it did not write.

## 8. What could not be measured

`ncu` does not run on this machine: `ERR_NVGPUCTRPERM`, GPU performance counters
are restricted to administrators. Everything above rests on `nsys` kernel tracing
and on timing prefixes of the layer walk, so there is no occupancy figure, no
stall-reason breakdown and no achieved-bandwidth counter for any single stage —
only wall-clock against known byte counts. The one place that limitation bites is
§5's second item, which is why it carries no number.
