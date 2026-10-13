# The SGLang prefill baseline — measured, and this is the table to beat

Taken on this machine, **on the same card this round ran on**, batch 1, history
0, prefill only (one prefill, sample one token, stop). **You do not need to
re-measure it.**

Two things are asked of the reader: measure your own side by "How it was
measured" below; and **decide for yourself which of the two columns is the line
comparable to yours**, and write the reason down.

## Two instruments, two columns

SGLang 0.5.21 · Qwen3.8-27B · TP 1 · radix cache off · prefill CUDA graph on
(the default). Measured 2026-10-12.

| prompt | **A: forward only** (median of 3) | range | **B: end to end** (median of 7) | range | compute floor |
|---:|---:|---:|---:|---:|---:|
| 1024 | **86.94 ms** | 0.56 ms | **89.06 ms** | 11.80 ms | 51.5 ms |
| 4096 | **319.94 ms** | 2.25 ms | **317.11 ms** | 37.43 ms | 208.6 ms |
| 8192 | **696.38 ms** | 0.71 ms | **638.25 ms** | 32.09 ms | 423.8 ms |

Figures derived from column B (instrument A does not report these):

| prompt | tok/s | TFLOP/s | of 989 | of peak at the clock | SM clock |
|---:|---:|---:|---:|---:|---:|
| 1024 | 11497 | 572.1 | 57.8% | 76.4% | 1500 MHz |
| 4096 | 12916 | 650.5 | 65.8% | 87.7% | 1485 MHz |
| 8192 | 12835 | 656.7 | 66.4% | 87.7% | 1500 MHz |

Raw numbers: `reports/sglang_prefill_baseline.json` (column B),
`reports/sglang_prefill_forward_only.jsonl` (column A).

### What each instrument is, and what is wrong with each

**Column A, forward only** — `sglang.benchmark.one_batch`, which goes around
the scheduler and calls the model runner directly, timing between
`synchronize()` calls. It measures one forward and **excludes** scheduling,
sampling, detokenizing and IPC. Each length runs in **its own process**, so the
one warmup inside it lands on the length being measured (`one_batch` warms only
at `input_len[0]` — run three lengths in one process and the last two measure
the **first** CUDA graph replay of that shape, which is unusable). Three
repeats per length.

Its flaw: it issues forwards back to back, at a tighter duty cycle than real
serving.

**Column B, end to end** — `sgl.Engine` down the full serving path, taking
`meta_info.e2e_latency` (measured server side). `max_new_tokens=1`, so the
number is TTFT. Per length, 2 untimed warmups and 7 timed, median.

Its flaw: it contains one cost your implementation does not have. Under the same
protocol a request that does almost nothing (4 tokens) takes **42.65 ms**, which
is the scheduler loop's fixed cost of waking from idle. **It has not been
subtracted** — subtracting it would put 1024 below its own compute floor
(51.5 ms), which is direct evidence that subtracting it is wrong.

### One contradiction, stated and deliberately not explained

At 8192, **A is 9.1% slower than B**. A forward is a proper subset of
end to end, so this cannot be explained by scope alone: the two instruments did
not measure the same machine state (most likely duty cycle → power → actual
clock; column A does not read the clock per repeat, so there is no direct
evidence). At 1024, A is 2.4% faster than B, as expected. At 4096, A is 0.9%
slower.

Each instrument is stable **against itself**: column A's ranges over three
lengths are 0.56 / 2.25 / 0.71 ms. So this is not noise; it is a systematic
difference between the two instruments.

**Which column is the line is the reader's call, with the reason written down.**
Pick one and nail it — not a different instrument per length.

## How it was measured, and why this protocol

```
forward only on both sides   the serving path's fixed costs count on neither.
                             When you measure yours: no weight loading, no
                             compilation, no tokenization, and be able to say
                             what the timer covers, from where to where

max_new_tokens = 1           one prefill, one sample, stop. No decode mixed in --
                             decode is memory bound and would dilute what is
                             being measured

disable_radix_cache          every repeat really recomputes the prefill. Left on,
                             the second repeat hits the cache and reports near
                             zero -- that is how a baseline goes quietly wrong by
                             two orders of magnitude. Every run asserts
                             cached_tokens == 0; it is not spot-checked

2 untimed warmups per length **cannot be skipped.** sglang 0.5.21 captures a CUDA
                             graph per prefill shape, and a shape's first replay
                             is far slower than the rest. Skip it and 8192 reads
                             20% slow (782.70 vs 629.95 ms)

then 7 timed, median         the range is reported beside the median
read the SM clock each time  right after the forward. Recorded on the row; it is
                             not used to correct anything
```

`bench_prefill.py` is in this directory and is the *definition* of the above.

Three assertions, any of which throws rather than reporting a number:
`cached_tokens == 0`, `completion_tokens == 1`, and every length
≤ `chunked_prefill_size`.

## This shape is compute bound — the reverse of the usual

`flops.py` counts FLOP from the safetensors headers rather than trusting the
parameter count on the model card.

| bucket | parameters | counted |
|---|---:|---|
| text matmul weights | 24.777 B | **yes**, `2 * n * P` |
| `embed_tokens` | 1.271 B | no — a table lookup is not a matmul |
| `lm_head` | 1.271 B | no — only the last position's logits, 1 row not n |
| vision tower | 0.461 B | no — this round feeds text |
| norm / bias / conv | 0.001 B | no — one-dimensional |

Attention scores are added for the **16 `full_attention` layers** only
(`2 * n² * d_attn` each, causal halving included). The other 48 are
`linear_attention`, O(n), and their projections are in the matmul bucket.

| n | matmul | scores | total | scores' share | compute floor |
|---:|---:|---:|---:|---:|---:|
| 1024 | 50.7 T | 0.2 T | **50.9 TFLOP** | 0.4% | 51.5 ms |
| 4096 | 203.0 T | 3.3 T | **206.3 TFLOP** | 1.6% | 208.6 ms |
| 8192 | 405.9 T | 13.2 T | **419.1 TFLOP** | 3.1% | 423.8 ms |

floor = FLOP / 989 TFLOP/s (H200 BF16 dense, no sparsity). **Re-check it with
`flops.py` before using it.**

That 8192 row: the weights are read once, 55.6 GB, to do 419.1 TFLOP —
arithmetic intensity about **7500 FLOP/byte**. The H200's ridge is **206**.
**36x past the ridge.**

So the floor here is a compute floor, not a traffic floor — an optimization
order sorted by "how many bytes were moved" is not sorting by the quantity that
binds.

## Why the lengths stop at 8192

`chunked_prefill_size` defaults to `None`, and sglang derives it from GPU memory
(`srt/arg_groups/memory_hook.py::_handle_gpu_memory_settings`):

| memory | card | `chunked_prefill_size` |
|---|---|---|
| < 90 GB | H100 / A100-80 | 8192 |
| **< 160 GB** | **H20 / H200** | **8192** |
| above | B200 / MI300 | 16384 |

The admission test in `managers/schedule_policy.py::add_one_req` is

```python
elif chunk_tokens_limit is not None and chunk_fit_tokens > chunk_tokens_limit:
    ...            # cut one chunk, leave the rest for later iterations
```

So 8192 goes through whole and 8193 is cut into two chunks across two
iterations. **8192 is the longest prompt sglang will take in one forward on this
card.** All three lengths are ≤ 8192, so all three rows are a single forward —
confirmed line by line in the scheduler log (`#new-token: 8192`, no chunking).

Two numbers that do **not** constrain these runs: `max_prefill_tokens = 16384`
only applies when chunked prefill is **off**; and `chunked_prefill_size` is a
budget for the **whole batch**, not per request, and an upper bound rather than
a target — one 100-token request in the queue runs a 100-token forward, not one
padded to 8192.

(Aside: with a prefix cache, only the **tail that missed** counts against the
budget, `cand_extend_input_len = len(full_untruncated_fill_ids) -
len(prefix_indices)`; attention still spans the full position range, since
chunking cuts the query axis and not the key axis. These runs start from an
empty cache, so prompt length and extend length are the same number.)

## What was turned on when measuring it

Everything sglang 0.5.21 turns on by default, with nothing switched off —
except the radix cache, which the method requires, and which only affects
prefill **reuse** rather than how fast one real prefill is.

| | |
|---|---|
| CUDA graph | decode and **prefill both on** (0.5.21 captures prefill graphs too) |
| precision | BF16, no quantization |
| TP | 1 |
| `chunked_prefill_size` | 8192 (stated explicitly; the same as what it derives) |
| speculative decoding | **off** — this round's implementation does not do it, and with it on the comparison is not one to one |

## The card and the clock: the thing that matters most on this machine

**Only one card is visible to this round, and it is pinned** — the one the two
columns above were measured on. `CUDA_VISIBLE_DEVICES` is already set.

Why pin it: three cards on this machine are locked to 1500 MHz, but **1500 is a
ceiling, not a floor**. The same 8192 work, measured on three of them (the
end-to-end instrument):

| | 8192 latency | SM clock | of peak at that clock |
|---|---:|---:|---:|
| card A | 696.07 ms | 1305 MHz | 92.4% |
| card B | 629.36 ms | 1500 MHz | 88.9% |
| card C | 637.08 ms | 1440 MHz | 91.5% |

**10.6% apart raw, 3.9% apart once normalised to each one's actual clock.** Each
card is repeatable against itself (two sessions a day apart on one card gave
637.08 and 638.25 ms, 0.19% apart); the spread is between cards, not within one.

What this means for a comparison: **the only comparison that holds between your
numbers and the two columns above is one made on the same card.**

Also: 989 TFLOP/s assumes the 1980 MHz boost, which is not reached here. So both
denominators are reported — `of 989` and `of peak at the clock`
(`989 * clock / 1980`). They differ enough to change how the table reads, which
is why neither is given alone.

## Re-checking it (not required)

```bash
python bench_prefill.py --ckpt "$QWEN38_CKPT" --lens 1024,4096,8192 --label recheck
# column A: one process per length, not three lengths in one process
python -m sglang.benchmark.one_batch \
    --model-path "$QWEN38_CKPT" --trust-remote-code \
    --batch-size 1 --input-len 8192 --output-len 1 --disable-radix-cache
# the FLOP account
python flops.py "$QWEN38_CKPT"
```

These run in an sglang environment, not this example's: sglang 0.5.21 · torch
2.13.0 · flashinfer 0.6.18 · triton 3.7.1 · sglang-kernel 0.4.7.

**A re-check does not replace the numbers above** — those were taken on the
pinned card in that session. A difference over 5% is worth reporting; that is
itself a result.
