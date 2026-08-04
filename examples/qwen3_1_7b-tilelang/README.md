# Qwen3-1.7B on TileFoundry — tilelang kernels

Verified at **v0.0.1**, 2026-08-04. Not verified since; nothing in CI re-runs it.

Greedy decoding of the published Qwen3-1.7B checkpoint taken end to end on
TileFoundry: every kernel on the path written here in tilelang, and the whole
decode step replayed from one captured CUDA graph.

    612.5 tok/s   one H200, batch 1, bf16, greedy, 2048 new tokens

---

## 1. Environment

Nothing here is installed by this directory; it is what the directory was written
and measured against.

| | |
|---|---|
| GPU | one NVIDIA H200, driver 575.57.08 |
| CUDA | 12.8 (`nvcc`), `CUDA_HOME` must be set |
| Python | 3.12 |
| `tilefoundry` | installed from the wheel, **not** editable — nothing here reads the source tree |
| borrowed from the environment | `torch` 2.9.1+cu128, `transformers`, `tokenizers` |
| extra package | `tilelang` 0.1.12 |
| weights | the published `Qwen/Qwen3-1.7B` checkpoint, 3.8 GB on disk |

## 2. How this was produced

One agent, one prompt, no human help after it started: Claude Opus 5 at xhigh
reasoning effort, 2.1 hours, 177 tool calls, no sub-agents.

The prompt was thirteen lines, where the other three examples here were produced
by one of about eighty. It is the Quick Start in the project's README, run
unedited:

```text
Get real tokens out of Qwen3-1.7B on TileFoundry, and make it fast.
Weights and config: <checkpoint directory>
Backend: tilelang.

Everything about TileFoundry is to be asked of the `tilefoundry` command -- do not
ask a person, do not go looking elsewhere. The model itself is yours to research.

Done when this runs from outside, prints the continuation, and reports a
tokens-per-second number measured over the whole generation:

    python run.py --prompt "Write a detailed explanation of how a GPU executes a matrix multiplication." --max-new-tokens 2048

Measure over a long generation -- 2048 new tokens, more than 2000 characters of
text. A 32-token sample is too short for the number to mean anything.
```

It names one file, `run.py`. That the work is a reference baseline first, then a
runtime twin, then `tilefoundry check` as the comparison between them — and the
shape of this directory — the agent read out of `tilefoundry tutorial` and decided
for itself.

## 3. How to use it

    python run.py --ckpt <checkpoint dir> --prompt "..." --max-new-tokens 2048

`--ckpt` is required: where the weights live is a fact about the machine, and a
default that exists on only one machine is a guess.

| flag | |
|---|---|
| `--prompt` | the text to continue; required |
| `--max-new-tokens N` | how many tokens to generate, default 2048 |
| `--device` | pin the runtime device. By default the emptiest visible one is taken, probed through Torch — an exclusive-mode card already has an owner, and only trying it says so |

The first run compiles the tilelang kernels (once, a few minutes).

    run.py                the entry point
    ref_src/              verbatim copy of the shipped `qwen3_1_7b` source -- the
                          reference, never edited
    fast/kernels.py       the tilelang kernels, one decode step's worth
    fast/engine.py        weight packing, buffers, and the graph capture
    fast/twin.py          @runtime_module twins, so `tilefoundry check` can judge
                          the kernels against the reference
    fast/test_kernels.py  the torch spelling of every kernel -- the interface they
                          were written against
    fast/arbitrate.py     an independent f64 reference, for the one disagreement
                          in §4 that `check` cannot settle

### Why it is shaped this way

The authored reference hands each step's key and value back for the caller to
`torch.cat` on. That is right for a reference — it keeps every shape expressed in
`ctx_len` alone — but the cache buffer then moves every step, and a CUDA graph
records addresses. So the engine takes the other form the migrate page names: a
cache of fixed capacity whose write window advances, with the position in a
one-element device tensor.

Everything a step needs then has a fixed address, so all 254 kernels are captured
once and replayed. The last kernel writes the sampled token back into the input
slot, and while the prompt still has a token left it feeds that one instead — so
one capture walks the prompt and continues past it with **no host round trip
anywhere in the loop**, including no sync to read the token back.

Decode is one token, so every projection is a GEMV: pure streaming, no reuse.
`q|k|v` and `gate|up` are packed into single matrices at load, because the cost of
a GEMV is the block count it can fill, not its arithmetic. Split-K supplies the
rest of the parallelism, and its f32 partials are reduced by the *consumer* — the
attention merge lands inside `o_proj`, and `silu(gate) * up` inside `down_proj`, so
neither is a launch of its own.

## 4. Where it stands

**Measured, on the environment above**, at four levels, each answering something
the level before it cannot.

1. **Every kernel against a torch statement of the same thing**, at production
   dimensions. All exact or within one bf16 rounding.
2. **`tilefoundry check` against the authored HIR** — the comparison the optimize
   page asks for. All four decoder-layer functions and all three root functions
   pass, at context extents 0 / 1 / 255 / 1024; and the whole model in one decode
   step passes on all 57 outputs (logits cosine 0.999951).
3. **An independent f64 reference**, because `check` says outright that a FAIL
   "proves disagreement, not which side is closer to truth". Where the twin and the
   reference differ on attention, the twin's error against f64 is **3.4e-3 vs the
   reference's 8.6e-3** — 2.5x closer.
4. **Against Hugging Face on the real checkpoint** — the L3 bar. Teacher-forced,
   **255/256 positions agree**.

The same authored HIR run through the evaluator instead of this twin decodes at
**14.8 tok/s**, so the twin is about 41x it.

### The two deliberate departures from the reference

Both move toward the published model, which is what level 3 and 4 measure:

- **The `1/sqrt(head_dim)` factor is applied after the dot product, in f32.** The
  reference multiplies it onto `q` in bf16 first, rounding every entry a second
  time; the exponential downstream turns that into percent-level error on the
  attention weights. HF scales after. This is most of the 2.5x in level 3. It is a
  fact about kernels that hold the score in bf16, not about the description: the
  authored HIR run through the Evaluator is unchanged either way, measured.
- **Attention probabilities are bf16 into the V product**, which is exactly what
  HF's attention does. Adopting it moved HF agreement from 253/256 to 255/256.

### The one remaining disagreement with HF

At the first generated token HF's bf16 logits for the two candidates are
**exactly equal** (22.625 and 22.625), so `argmax` picks the lower index. This
implementation keeps f32 logits, which resolve a real 0.11 gap, and picks the
other. Neither is wrong; HF's output dtype simply cannot represent the
distinction. Every other position in a 256-token teacher-forced comparison agrees.

### Three TileLang findings worth keeping

Each cost real time to locate and each is a cliff, not a gradient:

- **`from __future__ import annotations` breaks `@T.prim_func`.** Buffers are
  declared from parameter annotations, and PEP 563 hands the builder strings
  evaluated without the enclosing factory's closure — so the dimensions the
  factory exists to bind are exactly what fails to resolve.
- **`T.atomic_min` on shared memory costs 18 ms** where a `T.reduce_max` costs
  2 us. Argmax is written here as two max-reductions instead: the winning value,
  then `BN - j` over the entries attaining it, whose max is the lowest winning
  index — the same tie-break `torch.argmax` reports.
- **Reducing a GEMV accumulator in place costs two orders of magnitude.** Layout
  inference replicates the fragment across all threads to satisfy both uses, which
  spills it: 19 ms instead of 165 us. Staging through shared memory and reducing a
  fresh fragment fixes it. The same conflict has no workaround when the reduction
  is over the full hidden size, which is what stopped the residual-norm folds.

### Where the time goes

Marginal in-graph cost per decode step, at 1024 context:

| | per call | x | step | rate |
|---|---|---|---|---|
| `gate_up` GEMV | 14.1 us | 28 | 396 us | 3.6 TB/s |
| `down` GEMV (+silu) | 7.9 us | 28 | 221 us | 3.2 TB/s |
| `lm_head` (+argmax) | 165 us | 1 | 165 us | 3.8 TB/s |
| attention | 5.6 us | 28 | 156 us | — |
| `qkv` GEMV | 4.9 us | 28 | 138 us | 3.4 TB/s |
| `o` GEMV (+combine) | 4.3 us | 28 | 121 us | 1.9 TB/s |
| norms, rope | ~2 us | 84 | 165 us | — |

The GEMVs are at the streaming roofline — a sweep over block counts and split-K
factors found nothing better than 1% over the shapes in use, and block count
barely moves them. The remaining gap to the 3.44 GB / step memory floor is
per-kernel ramp and drain, so the only lever left is kernel *count*: the two
residual norms are single-block kernels costing ~1.9 us each of near-pure latency,
and folding them into the following GEMV is worth ~6-10% but is what the
layout-inference cliff above blocks.
