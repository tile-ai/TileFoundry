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
| handwritten, one cooperative launch | **231.3 tok/s** | 4.323 | **1.45x slower** |

Slower, throughout this document: the handwritten kernel takes 1.45 times as long
to produce a token as the TileLang one does. The goal was a handwritten kernel
**no slower** than TileLang, so the goal is not met.

It started this round at 6.510 ms and four changes took it to 4.323:

| change | ms/token |
|---|---|
| lane loops, `int4` reinterprets, eight hand-written partial sums | 6.510 |
| the same stages written against `ops::{dot, copy, tma_copy}` | 6.304 |
| the attention scan on `ops::mma` (§4) | 5.19 |
| the router's top-k folded over the block instead of one thread (§5) | 4.54 |
| two prefetch stages instead of three (§6) | **4.32** |

End to end through `bench_mine.py`, which is the number to quote. It is correct:
`compare_impls.py` puts the cooperative kernel at 1.5e-4 against the same stages
launched one at a time, with every state tensor bit-identical, and both against
the op-by-op reference at 3.4e-2 on the 5.0e-2 envelope `check_all.py` derives.

## 2. Where the time goes

Every stage can be switched off in the kernel (`NEMO_MEGA_SKIP`, a bitmask) and
its cost taken by difference; `stage_cost.py` does that and also switches off
whole branches, because a stage's own delta misses what it shares with its
neighbours. Turning *all* of them off leaves the layer walk and its barriers:
**0.744 ms of 4.242**, 17.5%.

| part | ms | share |
|---|---|---|
| the MoE branch, 23 layers | 1.965 | 46.3% |
| the Mamba branch, 26 layers | 1.289 | 30.4% |
| grid barriers and the launch | 0.744 | 17.5% |
| attention, 3 layers | 0.353 | 8.3% |
| the head | 0.112 | 2.6% |

and inside those, by stage:

| stage | ms | share | GB read | TB/s |
|---|---|---|---|---|
| MoE down x23 | 0.624 | 14.7% | 1.38 | 2.21 |
| Mamba in_proj x26 | 0.555 | 13.1% | 1.44 | 2.59 |
| MoE up x23 | 0.417 | 9.8% | 1.38 | 3.31 |
| Mamba out_proj x26 | 0.251 | 5.9% | 0.57 | 2.28 |
| Mamba ssm x26 | 0.238 | 5.6% | (state, not weights) | |
| MoE finish x23 | 0.204 | 4.8% | 0.46 | 2.25 |
| MoE shared_up x23 | 0.140 | 3.3% | 0.46 | 3.28 |
| attention scan x3 | 0.132 | 3.1% | | |
| MoE logits x23 | 0.121 | 2.9% | 0.02 | 0.13 |
| the head | 0.112 | 2.6% | | |

The whole step moves about 5.8 GB of weights in 4.24 ms: **1.37 TB/s**, against
TileLang's 1.96. Neither is near the card's 4.8, because a decode step reads
every weight exactly once and there is nothing for a cache to hold.

## 3. What the gap is made of

Two projections reach 3.3 TB/s and the rest sit at 2.2–2.6. The two fast ones
(`MoE up`, `MoE shared_up`) and the two slow ones (`MoE down`, `MoE finish`) do
the same amount of work over matrices of the same size; what differs is the row
length, which decides how wide a load the layout can give a lane:

| stage | row | per lane | move | TB/s |
|---|---|---|---|---|
| MoE shared_up | 2688 | 84 | 8 B | 3.28 |
| MoE up | 2688 | 84 | 8 B | 3.31 |
| MoE finish | 3712 | 116 | 8 B | 2.25 |
| MoE down | 1856 | 58 | 4 B | 2.21 |

The width is not the whole story either -- `finish` moves 8 B and is as slow as
`down` -- and a microbenchmark over the same bytes at four widths (`ops::dot`
over a 4096-wide row, forced to 1/2/4/8 elements a move) says the width is not
the lever at all:

| elements a move | TB/s |
|---|---|
| 1 | 0.63 |
| 2 | 2.11 |
| 4 | **2.27** |
| 8 | 1.20 |

Four is the peak and eight is *worse*, so widening the move is not what is
missing. That microbenchmark also puts a ceiling on the shape itself: one CTA
reading a tile, folding it, and reading the next reaches 2.27 TB/s and no more.
Every direct projection in the kernel is that shape, and every one of them
measures between 2.2 and 2.6 -- they are at the shape's ceiling, not below it.

**So the remaining gap is the shape, not the code inside it.** TileLang's 1.96
TB/s over the whole step against this kernel's 1.37 is what a deeper pipeline
buys, and closing it means restructuring how a CTA gets its tiles rather than
tuning a loop.

## 4. The attention scan: 21% of the step, for 3 layers of 52

The scan was a lane-per-key dot product: lane `l` took key `base + l`, computed
its own 128-element dot serially, and then a shuffle broadcast each key's weight
back so every lane could accumulate its slice of the value row. That is
`O(keys x 128)` scalar loads with a 32-way shuffle in the middle of it.

It is now the shape the TileLang kernel uses: `ops::mma` for a `(16, 128)` score
block, an online softmax over it with a query to sixteen adjacent lanes, and a
second `ops::mma` against the value block. `warp_reduce` grew a `Width` for the
per-query fold, so the row's reduction is a four-step butterfly over the sixteen
lanes that hold it rather than a shared round trip.

Three of 52 layers, and it was worth 21% of the whole step. The reason it is
worth so much more than its share of the layers suggests is that the old shape's
cost scales with the context and the new one does not.

## 5. MoE top-k: 11% of the step for 768 comparisons

Six passes of a 128-way scan, on one thread of one CTA, 23 times a step, with the
other 131 CTAs waiting at the grid barrier behind it. Measured 11.2% of the step.

The same six passes are now a butterfly over the 128 threads that already hold
the logits, on a packed `(score, expert)` key so one exchange carries both and a
tie still picks the lower index. It measures at the noise floor.

## 6. What was tried and did not work

| tried | measured |
|---|---|
| a third prefetch stage | 4.55 against 4.33 -- the loop does not consume the depth |
| staging Mamba's `in_proj` through shared memory | 4.554 against 4.538: no change |
| 16-byte moves instead of 8 | *slower*, see §3 |
| unrolling `ops::dot` with a compile-time trip count | 1.88 TB/s against 2.27: the register pressure costs more than the scheduling buys |
| moving all per-stage shared scratch into one arena | **23% slower**, see §7 |

## 7. The arena that should have been free

Every stage declared its scratch as `__shared__`, and the compiler lays those out
side by side across the whole call tree: 78 KB of static shared memory on top of
the 135 KB arena, which held the kernel to **one CTA an SM**. Carving them all
from the one arena instead dropped the static figure to 576 bytes and let two
CTAs sit on an SM.

It measured 23% *slower* at the same occupancy, and three targeted experiments
failed to say why:

- **aliasing**: `__restrict__` on the slice pointers and on every stage's
  parameters -- no change.
- **the L1/shared carve-out**: padding the arena back to the old 213 KB so the
  hardware makes the same split -- no change.
- **the address space**: the same `ops::dot` loop reading a static `__shared__`
  array against an `extern __shared__` arena, in isolation -- identical, 2.27
  TB/s both ways.

So the effect is real, reproducible, and unexplained. The static form is what
ships; the arena form is in the history for whoever picks this up.

## 8. A correctness bug the rewrite surfaced

The routed down projection gave each CTA a row count rounded up to whole tiles:
21 rows of `H` per CTA at 132 CTAs, rounded to 24. Each CTA therefore wrote three
rows its neighbour also wrote, and that stage accumulates with `atomicAdd`. Rows
near the end of the matrix were counted up to 14 times.

`check_moe.py` reported it the first time the rewritten kernel ran against it
(`acc` at 8.98e-1 on a 4.37e-3 bound). Splitting by `tile_span` -- disjoint runs
that cover the matrix -- fixes it. Worth naming because a store survives an
overlap and an accumulation does not, so the same shape was harmless in the up
projection two functions away.

## 9. Hypotheses that were wrong

Recorded because the wrong ones outnumbered the right ones, and each was
plausible enough to act on:

| guessed | measured |
|---|---|
| attention's placement is the biggest fixable item | attention is 8% of the step; its *implementation* was worth 21% |
| twelve expert launches are the cost | fusing them to two: 13% faster |
| shared-memory staging starves occupancy | removing it from MoE: no change |
| the accumulator dependency chain stalls the loads | eight accumulators: 3% faster |
| the router is fine, it reads 688 KB | it was 236 of the 319 us -- one thread per row, uncoalesced, one CTA |
| aliasing `__restrict__` on the in-place residual explains the drift | it does not; the drift is the expert accumulator's summation order |
| allocating 46 state tensors a step costs the missing millisecond | pooling them: no change |
| wider loads are faster | 16-byte moves measured *slower* than 8-byte ones |
| freeing 78 KB of shared memory to fit two CTAs an SM is free | 23% slower, §7 |

## 10. What is left

- **The pipeline shape (about 1.3 ms).** §3: every direct projection is at the
  ceiling of "read a tile, fold it, read the next" and TileLang is above it.
  This is the whole remaining gap and it is a restructuring, not a tuning.
- Not worth it: the barriers. 257 grid syncs at 2.9 us each is 0.744 ms, and
  merging stages to recover some of it costs the correspondence between a barrier
  and a reshard that the authored program states.

## 11. Token identity is a knife edge, not a gate

An earlier reading of this said the fused path had a fault of its own, because it
diverged from `transformers` where the stage-per-launch path did not. That
compared two different builds. Measured together — 48 steps, greedy,
teacher-forced, three prompts:

| prompt | `mega` (TileLang) | `cuda` (fused) | `cuda-stages` |
|---|---|---|---|
| `The capital of France is` | 48/48 | step 35 | step 35 |
| `In 1969 the first humans` | step 3 | step 3 | step 3 |
| `A prime number is` | step 42 | 48/48 | step 42 |

Every implementation loses it on some prompt. On the middle one all three diverge
at the same step and pick the same token as each other — a tie the reference
resolves one way and all three of these resolve the other. Nothing here separates
the handwritten kernels from the shipped one, and `README.md`'s "64 of 64" is the
first prompt with the TileLang kernel rather than a property any of them has in
general.

## 12. What could not be measured

`ncu` does not run on this machine: `ERR_NVGPUCTRPERM`, GPU performance counters
are restricted to administrators. Everything above rests on `nsys` kernel tracing
and on `stage_cost.py`'s difference timings, so there is no occupancy figure, no
stall-reason breakdown and no achieved-bandwidth counter for any single stage —
only wall-clock against known byte counts. That limitation is why §7 carries no
explanation.
