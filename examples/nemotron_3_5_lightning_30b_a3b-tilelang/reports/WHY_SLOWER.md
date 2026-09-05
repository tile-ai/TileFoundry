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

| | ctx 32 | ms/token |
|---|---|---|
| TileLang, one cooperative launch | **335.9 tok/s** | 2.977 |
| handwritten, one cooperative launch | **182.0 tok/s** | 5.495 |

1.85x off. Against the TileLang kernel the whole step lands at 4.513e-2 on the
5.018e-2 envelope `check_all.py` derives, with the same argmax at every step.
Token identity is where the fused path and the stage-per-launch path part: the
staged one matches `transformers` for 64 of 64 greedy steps, the fused one
diverges at step 35. They run the same device code, so that is an open fault of
the resident launch, not of the arithmetic, and it is not diagnosed here.

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

## 5. What is fixable and what is not

- **Attention's placement (fixable, ~1.3 ms).** Splitting by head at short
  context, the way the authored program does, would put the other 130 CTAs to
  work. The crossover `attention.py` already names is the number to dispatch on.
- **The shared-memory ceiling (fixable, unquantified).** A staged path that fits
  in a smaller ring, or a head that stages less, would let more CTAs per SM and
  lift the expert projections nearer their 74 us floor.
- **The barrier count (not worth it).** 5 barriers a layer at 2–3 us is about
  0.7 ms of the 5.49. Merging stages to save some of it costs the clarity of
  having one barrier per reshard the authored program states.

## 6. What could not be measured

`ncu` does not run on this machine: `ERR_NVGPUCTRPERM`, GPU performance counters
are restricted to administrators. Everything above rests on `nsys` kernel tracing
and on timing prefixes of the layer walk, so there is no occupancy figure, no
stall-reason breakdown and no achieved-bandwidth counter for any single stage —
only wall-clock against known byte counts. The one place that limitation bites is
§5's second item, which is why it carries no number.
