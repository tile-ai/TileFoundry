# SGLang prefill baseline worker: notes

Files (all in `bench/`):

| file | what it is |
|---|---|
| `sglang_worker.py` | the worker. Run with `ROOT/sglang-env/bin/python`. JSON lines on stdin/stdout. |
| `sglang_worker_selftest.py` | development driver: tokenizes English text, talks to the worker over the protocol, writes the JSONL. |
| `sglang_dev_run.jsonl` | raw replies of the development run below (load + 4 bench + quit). |
| `sglang_dev_run.log` | stderr of that run (SGLang logs + one summary line per T). |

Versions: SGLang 0.5.21, torch 2.13.0+cu130, flashinfer 0.6.18, sgl_kernel 0.4.7.
Model: `$QWEN38_CKPT` (`Qwen3_5ForConditionalGeneration`, 48 GDN + 16 gated full-attention layers, bf16).

## 1. SGLang entry points used

The worker uses the same in-process path as SGLang's own `bench_one_batch`
(in 0.5.21 the module is `sglang/benchmark/one_batch.py`; there is no
`sglang/bench_one_batch.py` in this install).

1. `ServerArgs.add_cli_args` + `ServerArgs.from_cli_args`, then the steps of
   `one_batch.main()`: `declare_resolution(... cuda_graph_config={"decode":{"max_bs":1}})`,
   `server_args.resolve_once()`, `_set_envs_and_config`, `PortArgs.init_new`.
2. The steps of `one_batch.latency_test()`: `publish(role="scheduler")`,
   `initialize_moe_config / fp8 / fp4`, `configure_logger`.
3. `one_batch.load_model(server_args, port_args, gpu_id=0, tp_rank=0)`. It builds
   `ModelRunner` (weights), `alloc_memory_pool`, `init_attention_backends`,
   the mamba SSU backend, `init_cuda_graphs` (prefill + decode graphs).
4. Per iteration, the steps of `one_batch.extend()` up to the forward:
   `Req(...)`, `ScheduleBatch.init_new(tree_cache=one_batch.TreeCacheNamespace(...))`,
   `batch.prepare_for_extend()`, `one_batch._maybe_prepare_mlp_sync_batch`,
   `input_ids` H2D, `ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)`.
5. The timed call: `ModelRunner.forward(forward_batch)`.
6. Freeing: `token_to_kv_pool_allocator.free(out_cache_loc)`,
   `req_to_token_pool.free_mamba_cache(req)`, `req_to_token_pool.free(req)`.

Server arguments (exact argv; everything else is the SGLang default):

```
--model-path $QWEN38_CKPT --tp-size 1 --disable-radix-cache --mem-fraction-static 0.5
+ decode cuda graph max_bs = 1 (bench_one_batch sets this to the batch size; decode is never run)
```

GPU memory of the worker: about 65-72 GB on the card (51 GB weights incl. vision
tower, KV + GDN-state pools, 2.6 GB prefill graphs). SGLang sizes the pools from
the free memory at start, so the pool size changes when other users hold memory
on the card (159,613 KV tokens / 59 GDN slots with an empty card, 104,809 / 39 in
the final run). This does not change a batch-1 prefill.

## 2. Backends that SGLang chose (default, H200 = SM90)

| part | choice | how it was found |
|---|---|---|
| attention backend (container) | `HybridLinearAttnBackend` | `model_runner.attn_backend` |
| full-attention layers (16) | FA3: `FlashAttentionBackend`, `fa_impl_ver=3`, `sgl_kernel.flash_attn` (`--attention-backend` resolved to `fa3`, prefill and decode) | backend object + log |
| GDN layers (48) | `GDNAttnBackend`; extend kernel `FlashInferGDNKernel` (log: "Defaulting SM90 GDN prefill backend to FlashInfer"); decode/verify kernel `TritonGDNKernel` | `kernel_dispatcher` + log |
| GDN causal conv1d | `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn` (profile: `_causal_conv1d_fwd_kernel`) | module + profile |
| GDN state | ssm state fp32 `[48 layers, slots, 48, 128, 128]`, conv state bf16 `[48, slots, 10240, 3]` | pool tensors |
| `--mamba-backend` | `triton` (decode SSU only) | resolved args |
| linear layers | `UnquantizedLinearMethod` (bf16); profile shows cuBLASLt `nvjet_sm90_*` GEMMs | `quant_method` + profile |
| KV cache | bf16, page_size 1, `HybridLinearKVPool` -> `MHATokenToKVPool` (16 layers, 4 KV heads x 256) | pool object |

## 3. CUDA graph and torch.compile facts for extend (prefill)

- Prefill CUDA graph IS ON by default: `cuda_graph_config.prefill.backend = "breakable"`
  (`default_prefill_backend()` returns `breakable` on CUDA; `Qwen3_5ForConditionalGeneration`
  is on `multimodal_breakable_cuda_graph_supported_model_archs`, so the multimodal rule does not
  disable it). Runner: `PrefillCudaGraphRunner` with `BreakableCudaGraphBackend`. It is kept
  at the default.
- What "breakable" does: the decoder body (64 layers + final RMSNorm) is captured as graph
  segments. The graph breaks at attention and GDN layers (`eager_on_graph` on
  `unified_attention_with_output` / `unified_linear_attention_with_output`), so FA3 and the GDN
  conv1d + delta-rule kernels run eagerly at replay with live metadata. Embedding and the
  LM head / logits processor run eagerly. No torch.compile.
- Capture buckets (tokens): 4..32 step 4, 48..256 step 16, 288..512 step 32, 576..1024 step 64,
  then up to 8192 (`chunked_prefill_size`). 128, 256, 512 and 1024 are exact buckets: no padding.
  Capture takes about 47 s at load.
- The worker asserts per iteration that the graph path ran: `ModelRunnerOutput.can_run_graph`
  is True for every timed iteration (`prefill_cuda_graph_used_per_iter`).
- `enable_torch_compile = False` (default). `tc_compiler="eager"` is only read by the
  `tc_piecewise` backend, which is not in use.
- Decode CUDA graph (`full`, bs=[1]) is captured by `load_model` but never replayed.

## 4. Exact timed region

```
torch.cuda.synchronize()
start_event.record(torch.cuda.current_stream())
out = model_runner.forward(forward_batch)          # ModelRunner.forward
end_event.record(torch.cuda.current_stream())
torch.cuda.synchronize()
ms = start_event.elapsed_time(end_event)
```

Inside `model_runner.forward` (read in the 0.5.21 source):

1. `_forward_raw` bookkeeping, `_prepare_eager_forward_batch` (no work for tp=1).
2. `_maybe_execute_deferred_mamba_cow_and_clear`: zeroes this request's fresh GDN state slot
   (48 layers, about 151 MB of fp32 ssm state). SGLang does this for every new request.
3. `PrefillCudaGraphRunner.execute`: `load_batch` (copies the live batch into static buffers),
   FA3 + GDN attention-metadata preparation, `model.forward`: embedding (eager), replay of the
   64-layer body + final norm (graph segments; eager FA3 and GDN kernels; K/V and GDN states are
   written to the pools here), `LogitsProcessor`: gather of the last position and the `lm_head`
   GEMM -> `next_token_logits [1, 248320]`.

So: lm_head and the last-position logits ARE inside. Attention-metadata preparation IS inside.
`ForwardBatch.init_new` and `ScheduleBatch.prepare_for_extend` are OUTSIDE.

Outside the timed region: tokenization (driver), Req / ScheduleBatch / ForwardBatch build,
KV-slot / req-slot / GDN-slot allocation, input_ids H2D, the NaN poisoning, argmax (no sampler
call), all assertions, freeing. `wall_ms` (perf_counter around the call + closing sync) is
reported as a cross-check; it is 0.03-0.04 ms above the event time.

## 5. How radix cache / prefix reuse is disabled

- `--disable-radix-cache` (resolved `disable_radix_cache=True`; this also turns off the mamba
  extra-buffer tracking, `enable_mamba_extra_buffer=False`).
- The batch uses `one_batch.TreeCacheNamespace`, a dummy tree cache with no prefix matching.
- Every iteration makes a new `Req` with a new rid. `req.prefix_indices` is empty. History = 0.
- All slots of the request are freed after each iteration. The free counts of the KV allocator,
  the req pool and the GDN-state allocator are asserted to equal the values before the
  iteration (no leak, nothing kept). The allocators give different slots to the next
  iteration (25 different first KV slots and 25 GDN slots per 25 iterations in the run).

## 6. Recompute evidence (assertions, every warm-up and timed iteration)

Batch metadata (asserted): `forward_mode == EXTEND`, `batch_size == 1`,
`extend_prefix_lens_cpu == [0]` and device `extend_prefix_lens == [0]`,
`extend_seq_lens == [T]` (cpu and device), `seq_lens == [T]`,
`len(input_ids) == len(out_cache_loc) == T`, `input_ids ==` the given token ids,
`positions == arange(T)`, every row of `mrope_positions == arange(T)`, `return_logprob == False`.

Poisoning (asserted):

- Before the forward: K and V at `out_cache_loc` in all 16 full-attention layers, and the
  request's GDN slot (conv state and ssm state, all 48 layers) are set to NaN. It is asserted
  that they hold NaN.
- After the forward: no NaN in any of these positions. Every GDN layer has a non-zero ssm state
  and a non-zero conv state (SGLang zeroes the slot inside the forward, so non-zero means the
  GDN kernels wrote it). K/V and the states agree with the first warm-up iteration
  (max abs diff <= 1% of the max magnitude; in the run they are bitwise equal).
- `next_token_logits` are finite; argmax equals the first warm-up argmax.
- K/V of layer 63 (a full-attention layer) are written for all T tokens. So all 64 layers ran
  over all T tokens in this forward.

Negative control (development check, not part of the worker): `model_runner.forward` was replaced
by a stub that returns the previous output without computing. The second iteration failed with
`AssertionError: NaN left in full-attention KV slots: K/V not written by this forward`.

Bench reply fields: `recompute_asserts` (all of the above), `logits_max_abs_diff` (max over the
vocabulary of max-min over all W+N iterations), `logits_max_abs_diff_vs_first_warmup`,
`logits_bitwise_equal_all_iters`, `argmax`, `warmup_argmax`.

## 7. Protocol

One JSON object per line in, exactly one JSON line out. Blank lines are ignored. fd 1 is
re-pointed to stderr at start, so SGLang prints cannot corrupt the protocol stream.

- `{"cmd":"load"}` -> `{"ok":true,"gpu_name","gpu":{"physical_index","uuid","pci_bus_id",...},"backends":{...},"load_s","memory",...}`.
  Optional: `model_path`, `mem_fraction_static`, `extra_server_args` (diagnostics only; the
  baseline uses none).
- `{"cmd":"bench","token_ids":[...],"warmup":W,"iters":N}` ->
  `times_ms[N]`, `median_ms`, `min_ms`, `max_ms`, `mean_ms`, `range_ms`, `warmup_times_ms[W]`,
  `wall_ms`, `argmax[N]`, `logits_max_abs_diff`, `prefill_cuda_graph_used`,
  `sm_clock_mhz{before,after,max,per_iter_after_forward[N]}`, `gpu_state{before,after}`
  (SM clock, other compute processes on the card, utilization, temperature), `gpu_physical_index`,
  `recompute_asserts{...}`. W must be >= 1 (warm-up 0 is the reference).
- `{"cmd":"quit"}` -> `{"ok":true,"cmd":"quit"}`, then exit.
- On any error (also a failed assertion): `{"ok":false,"cmd",...,"error","traceback"}`.

SM clock: NVML (`pynvml` is in sglang-env), handle by the UUID of `cuda:0`, so the physical index
is correct for any `CUDA_VISIBLE_DEVICES`. Read before and after each bench command and once
after each timed forward (outside the timed region).

## 8. Development numbers — development run, not the final table

Card: physical GPU 1 (H200, UUID GPU-3a3112ff-..., PCI 00000000:3B:00.0), `CUDA_VISIBLE_DEVICES=1`.
Session: 2026-10-15 00:53-00:54. Warm-up 5, timed 20, CUDA-event ms of `model_runner.forward`.
Prompt: an English paragraph tokenized with the model tokenizer, repeated and cut to T tokens.
All recompute assertions passed for all 25 iterations of every T. Logits, K/V and GDN states
were bitwise equal over the 25 iterations. Prefill CUDA graph used in every timed iteration.
11 compute processes of other users were resident on GPU 1 during the run (pytest-xdist and
python, ~16 GB, mostly idle).

| T | median ms | min ms | max ms | mean ms | SM clock MHz (before -> after; per-iteration min-max) | first warm-up ms | argmax |
|---:|---:|---:|---:|---:|---|---:|---:|
| 128 | 36.277 | 35.840 | 39.308 | 36.519 | 1500 -> 1500; 1500-1500 | 42.27 | 279 (" the") |
| 256 | 36.365 | 36.110 | 37.043 | 36.472 | 1500 -> 1500; 1500-1500 | 37.16 | 51624 (" keeper") |
| 512 | 48.653 | 48.547 | 56.707 | 49.200 | 1500 -> 1500; 1500-1500 | 54.20 | 264 (" a") |
| 1024 | 85.914 | 85.043 | 87.457 | 86.081 | 1500 -> 1500; 1440-1500 | 90.60 | 42046 (" storms") |

The JSONL was written by the final worker code except one cosmetic change made later (format of
`gpu.pci_bus_id`; the JSONL shows `00000000:3b:00`). A check run of the final file after that
change (T=128, warm-up 2, timed 3) gave median 36.233 ms, all assertions passed.

An earlier run with the same forward path (same card, about 00:49, 0 other processes on GPU 1)
gave medians 35.606 / 35.897 / 48.835 / 85.940 ms (min 35.318 / 35.639 / 48.811 / 85.259,
max 39.617 / 36.900 / 49.060 / 88.537). The two runs agree within 2%.

## 9. Observations that matter for the comparison

- At T=128 and T=256 the SGLang forward is host-bound, not GPU-bound. The two medians are
  almost equal (36.3 vs 36.4 ms). A torch.profiler run of one T=128 forward gave 21.6 ms of GPU
  kernel time in a forward that takes 36.4 ms without the profiler. The host time is in the eager
  GDN break (`sglang::unified_linear_attention_with_output`: about 0.9 ms CPU per GDN layer under
  the profiler, 48 layers; aten ops and Python / CuTe-DSL launch work). At T=1024 the kernel sum
  is 82.9 ms of an 85.4 ms forward: GPU-bound. These are SGLang's own forward costs, not
  serving-path costs.
- Informational only (not the baseline): with `--linear-attn-prefill-backend triton` (Triton GDN
  extend kernel instead of FlashInfer) the medians were 36.67 / 36.82 / 48.03 / 88.14 ms. So the
  short-T floor is not caused by the FlashInfer GDN kernel.
- The first warm-up iteration of a new T is 3-6 ms slower (table above). The later warm-ups
  are at the steady value.
- The SM clock is at 1500 MHz (the cap on this card) and drops to 1440-1455 MHz during T=1024.
- `gpu_state.other_compute_procs` in each reply records how many other processes share the card.
