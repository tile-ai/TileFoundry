# The SGLang baseline — measured, and this is the table to beat

Taken on this machine, one H200, batch 1, **decode only**. You do not need to
re-measure it; if you want to check it, the method and the script are below.

## decode tok/s vs context length

| context | 0 (32) | 1K | 4K | 16K | 32K | 64K | 128K | 256K (262080) |
|---|---|---|---|---|---|---|---|---|
| **tok/s** | **297.0** | 295.7 | 294.7 | 292.7 | 291.2 | 286.7 | 292.0 | **258.4** |
| ms/token | 3.366 | 3.382 | 3.393 | 3.417 | 3.434 | 3.488 | 3.425 | 3.870 |

Spread over three measurements: 0.000–0.055 ms, except the 128K and 256K points
at 0.363 / 0.501 ms, which run 32 steps rather than 256.

**The shape of this table matters more than its values.** Context grows 8000x
and decode loses 13%. Twenty-three of the fifty-two layers are Mamba2, whose SSM
state does not grow with context; only six are full attention, and their KV is
1.6 GB even at 262K. So this model's decode is close to a flat line, and **at
every length you are up against something near 290 tok/s** — there is no free
ground to be had from an opponent that falls over at long context.

## TTFT, measured in passing (not a criterion here)

| context | 32 | 1K | 4K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|---|
| TTFT (ms) | 26 | 33 | 164 | 620 | 1243 | 2662 | 5921 | 14734 |

Prefill is not part of this work; this row only says how much time sits in front
of each point of the table above.

## How it was measured, and why this way

**Decode only — not "decode with the prefill subtracted out".**

```
radix cache off  ->  run(n, 1) and run(n, 1+S) both certainly recompute prefill
decode = (t(1+S) - t(1)) / S     what the subtraction leaves is S decode steps
S = 256 (<128K) / 32 (>=128K), three measurements per point, median
time from meta_info.e2e_latency (server side), not the client's wall clock
```

Three assertions, any of which throws rather than reporting a number:
`cached_tokens == 0` (neither run hit the cache, which is what makes the
subtraction valid), the generated step counts match, and the difference is
positive.

**Turning the cache off does not weaken the baseline**: the radix cache only
affects prefill reuse, and decode speed does not depend on it.

## What was turned on

sglang 0.5.16's defaults, with nothing switched off except the radix cache,
which the method above requires:

| | |
|---|---|
| attention | `fa3` |
| MoE runner | `flashinfer_cutlass` |
| sampling | `flashinfer` |
| CUDA graph | decode `full` / prefill `breakable`, on throughout |
| overlap schedule | on |
| mamba | `triton` (tried `flashinfer`, **0.5% slower** — below) |
| linear attn | `triton` |
| precision | BF16, unquantized |

Tried and not taken:

| configuration | 1024 | 32768 | verdict |
|---|---|---|---|
| baseline (triton mamba) | 295.71 | 291.24 | **used** |
| `mamba_backend=flashinfer` | 294.40 | 289.58 | 0.4–0.6% slower |
| `enable_torch_compile` | — | — | **would not compile**, below |

`enable_torch_compile` was still compiling after fifteen minutes when it was
killed, with the log filling with
`ttir analysis hit an op we do not know how to analyze: tt.elementwise_inline_asm`
— inductor cannot analyze sglang's triton kernels that carry inline PTX. **There
is no number for this one**; the two blanks above mean "not measured", not
"measured and no different". If you can make it compile and it is faster, the
baseline should go up.

**No speculative decoding.** This model carries
`num_nextn_predict_layers=1` and sglang has `nemotron_h_mtp`, so turning it on
would make SGLang faster — but the implementation this is compared against does
not do MTP, and then it would not be the same thing being measured. Off on both
sides.

## Re-measuring

```bash
CUDA_VISIBLE_DEVICES=0 python bench_sglang.py \
    '{"mem_fraction_static":0.85,"disable_radix_cache":true}' \
    '32,1024,4096,16384,32768,65536,131072,262080' out.json
```

Environment: sglang 0.5.16 · torch 2.11.0+cu130 · flashinfer 0.6.14 ·
triton 3.6.0 · H200 at 1500 MHz.

## Four things sglang does on this model that you will hit too

Written down because anyone re-measuring will walk into them in the same order:

1. **`bench_one_batch` asserts on this model** —
   `Mamba selective_state_update backend not initialized`. The only call to that
   initializer in the tree is in `srt/managers/scheduler.py`, and
   `bench_one_batch` goes around the scheduler. So it has to be the server or
   `sgl.Engine`.
2. **The HTTP server plus `bench_serving` hits two environment problems** — a
   SOCKS proxy set in the environment with `socksio` not installed in httpx, and
   a warmup that wants to download ShareGPT by default. Starting `sgl.Engine`
   in-process has neither.
3. **The first subtraction produced negative numbers** — the radix cache had the
   prefix, so the second run's prefill was nearly free and `t(1+S) < t(1)`.
4. **Skipping prefill *via* the cache does not work either** —
   `mamba_track_interval=256`, so an SSM checkpoint exists only every 256 tokens;
   below that there is nothing to reuse and `cached_tokens` is 0, and in between
   whether it hits depends on alignment. So prefill was sometimes subtracted and
   sometimes not. The fix was the other way round: **turn the cache off so both
   runs are symmetric.**

The first three attempts produced, in order: negative numbers, −1397 tok/s, and
701 tok/s at 4096 — that last one is above the roofline floor of 580 tok/s, which
is the direct evidence it could not be trusted. **The three assertions above were
added against exactly those three failure modes.**
