# Qwen text prefill on TileFoundry — Apple AMX kernels

Runtime execution and CLI integration rechecked on **2026-10-19**, in the
development checkout based on `3cf660e`.

Qwen3-1.7B and Qwen3.5-4B text prefill on one M2 Pro: token IDs in, last-position
logits out. Batch one, no history. The large projections use FP16 AMX outer
products with FP32 accumulation; attention, GDN and the other host operations
remain in the supplied native llama.cpp runtime. The model, its projection schedules and its runtime are all present
in this directory.

Section 4 compares this AMX runtime with local llama.cpp Q8_0 and F16 execution.

---

## 1. Environment

This is the environment used for the current execution checks. The entry point
uses the installed packages and builds its FFI entry on first use.

| | |
|---|---|
| Machine | Apple M2 Pro, 32 GiB unified memory |
| Execution | CPU and Apple AMX, eight logical workers |
| Python | 3.12.13 |
| `tilefoundry` | this development checkout, commit `3cf660e` |
| Torch | 2.13.0, for resources and FFI tensors |
| Native model runtime | the supplied experimental llama.cpp checkout, selected by `LLAMA_CPP` |
| Other packages | `gguf` 0.19.0, `safetensors` 0.8.0, `apache-tvm-ffi` 0.1.12 |
| Native toolchain | CMake and Xcode Command Line Tools; `clang++` and SDK located through `xcrun` |
| Weights | Qwen3-1.7B or Qwen3.5-4B, Q8_0 GGUF |

Q8_0 weights are dequantized to FP16 during preparation. That preserves the
checkpoint's quantization error and changes the projection arithmetic from
the native Q8 path. Prepared weights take more memory than the GGUF file.

## 2. How this was produced

The task for this example, stated as a prompt:

```text
Get Qwen3-1.7B and Qwen3.5-4B text prefill running on TileFoundry.
Weights: the Q8_0 GGUF checkpoints.
Backend: Apple AMX on M2 Pro, eight workers.
Given: the existing user-provided AMX runtime, including its native
instruction wrappers and projection implementation. Use it as the
starting point for the C++ runtime and AMX instruction contracts.
Preserve its existing model execution path and host operations.
Scope: batch one, token IDs in, last-position logits out.

Deliver model.py (whole-model HIR), scheduled.py (scheduled HIR),
runtime_model.py (Python runtime modules), kernels/ (C++ AMX kernels),
and run.py (the entry point).

Make the weight layouts, AMX instruction choices and worker partition
explicit in the scheduled HIR. Put the AMX instruction support in
TileFoundry itself. Both HIR modules must be usable through the CLI.

The runtime module should load prepared weights and execute directly.
Build the native FFI entry on first use against the supplied runtime. Verify execution without rerunning
performance experiments or adding tests.
```

## 3. How to use it

From the repository root:

```sh
export LLAMA_CPP=/path/to/the/supplied/llama.cpp
python examples/qwen_prefill-amx/run.py --model 1.7b --checkpoint /path/to/Qwen3-1.7B-Q8_0.gguf
python examples/qwen_prefill-amx/run.py --model 4b --checkpoint /path/to/Qwen3.5-4B-Q8_0.gguf
```

The first run prepares packed projection weights under `prepared/<model>/`
beside `run.py`, builds the supplied native libraries and FFI entry under
`kernels/.build/`, and loads the model. Subsequent runs reuse the outputs.
`LLAMA_CPP` must contain the provided AMX implementation and its
`tf_amx_load_prepared` / `tf_amx_clear_prepared` exports; a standard upstream
checkout alone does not provide that execution path. `LLAMA_BUILD` can select
an existing CPU/Accelerate build. No source checkout is modified automatically.
The projection kernel is in the supplied `ggml/src/ggml-blas/shared_fp16.hpp`;
`kernels/prefill.cpp` provides its FFI and native model-driver bridge.

`--prepared` can reuse the original experiment's packed `.bin` directory
for the same checkpoint; its metadata does not need a separate manifest.

| Flag | Meaning |
|---|---|
| `--model 1.7b\|4b` | Which model to construct; required |
| `--checkpoint PATH` | Source Q8_0 GGUF; required |
| `--prepared DIR` | Prepared resource directory; defaults to this example's `prepared/<model>/` |
| `--sequence N` | Static prompt length, a multiple of 32 from 128 to 8192; default 128 |

`run.py` feeds a fixed valid token sequence through one prefill and prints the
logits shape and next-token ID. It stops there: the example covers text prefill,
with no decode loop, MTP or vision path.

```text
model.py                 the whole-model HIR and checkpoint weight converters
scheduled.py             the same model with explicit AMX projection schedules
runtime_model.py         RuntimeModule adapter for the original native model driver
kernels/prefill.cpp       the original TVM FFI model and projection entries
kernels/__init__.py       first-use native build and load
weights.py               GGUF reading and checkpoint-to-module weight aliases
config.py                the two models' architecture parameters
run.py                   preparation, loading and one runtime call
```

The runtime also accepts caller-supplied token IDs. After preparing the weights,
run this from the example directory:

```python
import torch
from config import config
from runtime_model import QwenPrefill
from weights import PreparedModelResource

resource = PreparedModelResource("/path/to/Qwen3-1.7B-Q8_0.gguf", "prepared/1.7b")
with QwenPrefill(config("1.7b"), sequence=128) as runtime:
    runtime.load(resource)
    token_ids = torch.arange(128, dtype=torch.int32)
    logits = runtime(token_ids)
```

`model.build(c, 128)` constructs the authored HIR;
`scheduled.make_scheduled(c, 128)` constructs the scheduled HIR used by
`runtime.module` for inspection. Execution calls the original native model
graph rather than evaluating the IR. Use one checkpoint per process, as in
the supplied runtime. The HIR keeps weights on the root module and passes them
explicitly to shared child definitions, retaining the root's weight converters.
Each source file also exposes `Qwen3_1_7B` and `Qwen3_5_4B` as
CLI-selectable modules with a 128-token prompt:

```sh
tilefoundry analyze examples/qwen_prefill-amx/model.py:Qwen3_1_7B /tmp/qwen-hir.txt --compute-cost --memory
tilefoundry schedule candidates examples/qwen_prefill-amx/model.py:Qwen3_1_7B /tmp/qwen-candidates.txt
tilefoundry analyze examples/qwen_prefill-amx/scheduled.py:Qwen3_1_7B /tmp/qwen-scheduled.txt --compute-cost --memory
tilefoundry schedule finalize examples/qwen_prefill-amx/scheduled.py:Qwen3_1_7B /tmp/qwen-tir.py
```

To generate the modeled performance timeline:

```sh
tilefoundry analyze examples/qwen_prefill-amx/scheduled.py:Qwen3_1_7B /tmp/qwen-performance.txt --performance
```

Use `:Qwen3_5_4B` for the hybrid model. Analysis and finalization run on the
model descriptions and do not require weights or the native extension.

## 4. Where it stands

Measured on **2026-10-19**, Apple M2 Pro, 32 GiB unified memory, CPU-only, eight
threads/workers. Our implementation uses the supplied packed-FP16 AMX runtime
for eligible large projections and retains its native C++ model execution.
The AMX column measures that modified llama.cpp execution path; the model
driver and host operations still come from llama.cpp.
llama.cpp is the local `e5819f0dd` working tree (build 2, existing BLAS/AMX edits),
rebuilt in Release with AppleClang 21.0.0. Accelerate, BLAS, native ARM
instructions, CPU repacking and Flash Attention are enabled; Metal and KleidiAI
are disabled. Native baselines unset `TF_AMX_MODE` / `TF_AMX_CHECKPOINT`, allow
BLAS dispatch (`-nopo 0`) and disable thread-pool polling (`--poll 0`).

CPU builds with and without OpenMP were checked. One configuration was selected
per model and weight format by the lower sum of medians across all four lengths,
then retained for that entire group. Qwen3-1.7B Q8_0 uses the native thread pool;
the other native groups enable OpenMP in the CPU and BLAS backends. The local
CMake configuration required an explicit `GGML_USE_OPENMP` definition and
OpenMP linkage on the BLAS target to activate its existing parallel-conversion
branch; the llama.cpp C++ source was unchanged. OpenMP uses eight threads,
passive waiting and `KMP_BLOCKTIME=0`; all runners set
`VECLIB_MAXIMUM_THREADS=8`.

Batch-one text prefill, zero history, last-position logits, batch/microbatch
2048 and F16 K/V cache. Each length has one full-length warmup and five timed
calls, run serially. All latencies below are **median milliseconds**. Preparation,
compilation and loading are excluded. Our timer includes state reset, FFI and
final-logit copy; llama-bench clears state before its timer. No decode, MTP or
vision is timed. CPU affinity and clocks were not locked.

| Model | Prompt tokens | Our AMX (ms) | llama.cpp Q8_0 (ms) | llama.cpp F16 (ms) | vs Q8_0 | vs F16 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 128 | 192.90 | 220.18 | 368.14 | 1.14× | 1.91× |
| Qwen3-1.7B | 512 | 741.72 | 901.61 | 988.67 | 1.22× | 1.33× |
| Qwen3-1.7B | 1024 | 1655.07 | 2025.08 | 1827.90 | 1.22× | 1.10× |
| Qwen3-1.7B | 2048 | 3711.20 | 4742.70 | 3732.24 | 1.28× | 1.01× |
| Qwen3.5-4B | 128 | 525.31 | 684.32 | 1107.83 | 1.30× | 2.11× |
| Qwen3.5-4B | 512 | 2085.52 | 2743.58 | 2727.69 | 1.32× | 1.31× |
| Qwen3.5-4B | 1024 | 4126.70 | 5899.10 | 5241.54 | 1.43× | 1.27× |
| Qwen3.5-4B | 2048 | 9653.82 | 11808.71 | 10596.87 | 1.22× | 1.10× |

Ratios are **llama.cpp latency / our AMX latency**: above 1× means our runtime
is faster. Results close to 1× indicate similar performance within the limits
of this desktop measurement.

All paths start from the same Q8_0 GGUF checkpoints. Our runtime prepares FP16
projection weights; native F16 files are expanded offline using
`llama-quantize --allow-requantize ... F16 8`, preserving source quantization
error and auxiliary F32 tensors. Native BLAS-selected F16 matmuls convert
weights to F32 and call `cblas_sgemm`.

The measured entry dispatches through the supplied AMX backend: each prefill
has 193 AMX matrix calls for 1.7B and 197 for 4B. Across all eight measured shapes,
its logits match the archived experiment outputs exactly (maximum absolute
difference 0). The table uses these current-entry measurements. First-use
weight preparation and execution were also checked on 1.7B.
