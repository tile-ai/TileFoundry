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
8](#8-token-identity-is-a-knife-edge-not-a-gate). Speed is the whole of what is
wrong here.

## 2. Where the time goes

Every stage can be switched off in the kernel (`NEMO_MEGA_SKIP`, a bitmask) and
its cost taken by difference. Turning *all* of them off leaves the layer walk and
its barriers: **0.34 ms of 5.48**, so the resident structure is 6% and not the
problem. The rest:

| stage | of the step | per layer | share |
|---|---|---|---|
| MoE down x23 | 1.298 ms | 56.4 us | 23.7% |
| MoE up x23 | 0.943 ms | 41.0 us | 17.2% |
| Mamba in_proj x23 | 0.708 ms | 30.8 us | 12.9% |
| MoE top-k x23 | 0.524 ms | 22.8 us | 9.6% |
| Mamba ssm x23 | 0.350 ms | 15.2 us | 6.4% |
| MoE finish x23 | 0.296 ms | 12.9 us | 5.4% |
| MoE shared_up x23 | 0.285 ms | 12.4 us | 5.2% |
| Mamba out_proj x23 | 0.232 ms | 10.1 us | 4.2% |
| head | 0.161 ms | 160.6 us | 2.9% |
| everything attention x6 | 0.227 ms | | **4.1%** |
| the rest | 0.111 ms | | 2.0% |

An earlier version of this document said attention was the largest fixable item
at 28% of the step. That came from differencing prefixes of the layer walk, which
attributes badly. Attention is 4%.

## 3. The cause: how long a run of tiles each CTA gets

Every stage here is a matrix-vector product over a large matrix, so the step is
bandwidth-bound throughout and the only question is what fraction of the card's
bandwidth each stage reaches. Against the bytes each one has to move:

| stage | MB | us | TB/s | tiles a CTA gets | staged? |
|---|---|---|---|---|---|
| head | 704.6 | 160.6 | **4.39** | 124.1 | yes |
| attn o_proj | 22.0 | 9.9 | 2.22 | 2.5 | no |
| Mamba out_proj | 22.0 | 10.1 | 2.18 | 2.5 | no |
| Mamba in_proj | 55.4 | 30.8 | 1.80 | 9.8 | yes |
| MoE shared_up | 20.0 | 12.4 | 1.61 | 3.5 | no |
| MoE finish | 20.0 | 12.9 | 1.55 | 2.5 | no |
| attn qkv | 24.8 | 16.8 | 1.47 | 4.4 | no |
| MoE up | 59.9 | 41.0 | 1.46 | 1.8 | no |
| MoE down | 59.9 | 56.4 | **1.06** | 1.8 | no |

Whole step: TileLang 2.35 TB/s, this kernel 1.28 TB/s.

**The head reaches 4.39 TB/s.** Same code, same card, same kernel: the hardware
and this implementation can go near peak. What the head has that nothing else
does is 124 tiles per CTA -- a long enough run for the three-deep prefetch to
have anything to hide behind. The two MoE expert projections have 1.8, the
fewest of any stage, and they are 41% of the step.

The lever is the run length, not the launch structure and not the placement. 232
tiles split over 132 CTAs is under two apiece; the prologue issues three and the
loop consumes two, so the pipeline never starts. Treating the six experts as one
stream of 1392 tiles would put ten tiles in each CTA, which is what Mamba's
in_proj has, and in_proj reaches 1.80 TB/s.

## 4. What that does not explain

Ten tiles a CTA is 1.80 TB/s, not 4.39. Bringing every projection to in_proj's
figure would take the step to roughly 4.7 ms -- 214 tok/s against TileLang's 336.
So run length is the measured lever and it is not the whole gap; what else
TileLang does per stage has not been established here.

## 5. MoE top-k costs 9.6% and does almost nothing

768 comparisons, on one thread of one CTA, 23 times a step. 22.8 us a layer, and
131 CTAs sit at the barrier behind it. This one is not bandwidth, it is a
serial section that was never sized against what waits for it.

## 6. Hypotheses that were wrong

Recorded because the wrong ones outnumbered the right one four to two, and each
was plausible enough to act on:

| guessed | measured |
|---|---|
| attention's placement is the biggest fixable item | attention is **4%** of the step; the guess came from differencing layer prefixes, which attributes badly |
| twelve expert launches are the cost | fusing them to two: **13% faster** |
| shared-memory staging starves occupancy | removing it from MoE: **no change** |
| the accumulator dependency chain stalls the loads | eight accumulators: **3% faster** |
| the router is fine, it reads 688 KB | it was **236 of the 319 us** — one thread per row, uncoalesced, one CTA |
| aliasing `__restrict__` on the in-place residual explains the drift | it does not; the drift is the expert accumulator's summation order |
| allocating 46 state tensors a step costs the missing millisecond | pooling them: **no change** |

Only `nsys` and the layer-prefix timings moved this forward; every hypothesis
formed by reading the code was wrong.

## 7. What is fixable

- **Run length on the MoE experts (1.6 ms of 5.5).** Six experts as one stream of
  1392 tiles instead of six streams of 232 puts ten tiles in a CTA. At in_proj's
  1.80 TB/s that is 33 us a projection instead of 41 and 56.
- **Top-k (0.5 ms).** Anything that is not one thread of one CTA.
- **Whatever takes 1.80 TB/s to 2.35 (about 1 ms).** Not established. The head
  shows 4.39 is reachable, so the ceiling is not the card.

Not worth it: the barriers. Five a layer at 2-3 us each is 0.34 ms, and merging
stages to recover some of it costs the correspondence between a barrier and a
reshard the authored program states.

## 8. Token identity is a knife edge, not a gate

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

## 9. What could not be measured

`ncu` does not run on this machine: `ERR_NVGPUCTRPERM`, GPU performance counters
are restricted to administrators. Everything above rests on `nsys` kernel tracing
and on timing prefixes of the layer walk, so there is no occupancy figure, no
stall-reason breakdown and no achieved-bandwidth counter for any single stage —
only wall-clock against known byte counts. The one place that limitation bites is
§5's second item, which is why it carries no number.
