# Qwen text Prefill on Apple AMX

Qwen3-1.7B and Qwen3.5-4B, batch one, text Prefill only. The example provides
whole-model HIR, explicit AMX schedules, retained-call TIR and a hand-written
TVM FFI runtime. No Decode, MTP, vision or compiler code generation is involved.

The model uses `ModuleList` instance slots: the dense model's 28 layers share
one layer definition; the hybrid model's 32 layers share two. Each instance
retains its own resources. Static calls select the heterogeneous layer slots;
the compiler does not clone or inline their bodies.

## Files

| File | Role |
|---|---|
| `model.py` | Dense/hybrid architecture and whole Prefill HIR |
| `scheduled.py` | Explicit destinations, activation/weight layout and worker schedule |
| `instructions.py` | Example-local X/Y load, FMA16, Z store and packing contracts |
| `host_instructions.py` | Typed host software kernels for non-AMX operations |
| `runtime_model.py` | One runtime twin per shared HIR definition |
| `weights.py` | GGUF aliases and Q8_0 decoding |
| `native/` | Standalone hand-written AMX kernel, thread pool and reverse-engineered instruction header |
| `prepare.py`, `run.py`, `export_ir.py`, `check.py` | Reproduction entry points |

AMX contracts are local extensions in this example, not claims that an AMX
code generator has been added to TileFoundry. Host operations are software
kernels implemented using Torch; they are not AMX ISA instructions. The native
projection kernel and its FFI module have no llama.cpp dependency.

## Schedule and numerical contract

Weights arrive as Q8_0 GGUF and are dequantized to FP16. The original quantization
error remains. `Module.prepare` packs each projection from `[N,K]` to `[N/B,K,B]`,
where B is 64, or 32 for narrow projections. Activations are converted to FP16
and packed into contiguous 32-row panels before accumulation.

Eight logical workers distribute N panels cyclically. Each holds a 32x32 FP32
Z accumulator, advances K by loading 32 weights into X and 32 activations into
Y, and issues FP16 outer products with FP32 accumulation. Z stays resident until
the tile is complete, then its interleaved rows are unpacked. The final M=1
vocabulary projection uses a host kernel; it is not a Decode workload.

The reverse-engineered AMX header is retained from the original user-provided
runtime. This path is validated on M2 Pro. Worker identity is logical; the
runtime does not promise OS thread affinity.

## Run

Use an Apple Silicon Mac with Command Line Tools and the existing `tf` Conda
environment. Dependencies are TileFoundry from this checkout, Torch, `gguf`,
`safetensors` and `tvm_ffi`. Use the same Python for building and running.

```sh
conda activate tf
python examples/qwen_prefill-amx/build.py
python examples/qwen_prefill-amx/check.py

python examples/qwen_prefill-amx/prepare.py --model 1.7b --checkpoint /path/to/Qwen3-1.7B-Q8_0.gguf
python examples/qwen_prefill-amx/run.py --model 1.7b

python examples/qwen_prefill-amx/prepare.py --model 4b --checkpoint /path/to/Qwen3.5-4B-Q8_0.gguf
python examples/qwen_prefill-amx/run.py --model 4b
```

Config snapshots are in `configs/`. Checkpoint paths must be supplied explicitly;
there are no user-specific absolute paths or downloads during a run. `prepare`
uses the authored converters and writes canonical weights under
`artifacts/prepared-<model>`. `run` loads only those prepared resources and
executes one 128-token Prefill over a fixed valid token pattern. It checks the
logits shape and finiteness, prints the first-token argmax and actual AMX call
counts, and saves `smoke.json`. It has no warmup, repetition or speedup table.

Prompt lengths must be positive multiples of 32; `--sequence` selects the static
shape. The prepare/load/run stages are separate, and packed weight layouts do
not depend on prompt length. Large prepared weights and generated binaries are
ignored by Git.

## Inspect the compiler path

```sh
python examples/qwen_prefill-amx/export_ir.py --model 1.7b
python examples/qwen_prefill-amx/export_ir.py --model 4b
python examples/qwen_prefill-amx/verify_ir.py --model 1.7b
python examples/qwen_prefill-amx/verify_ir.py --model 4b
```

The three source files are emitted into `artifacts/ir-<model>/`. Finalization
uses the current API and retained calls, with no old node-budget, recursion-limit
or no-traffic workaround. The runtime is hand-written; emitting TIR does not
compile that TIR into the executed native kernel.

`check.py` is a small dense/hybrid migration check: actual converters, loaded
resources and native AMX are compared with the unscheduled whole-model evaluator.
The full-model smoke confirms execution, not generation quality. This migration
does not rerun performance experiments and does not claim the old llama.cpp
bridge's speedups for this runtime.

## Execution validation

Rechecked on M2 Pro on 2026-10-19 using the commands above and existing prepared
Q8_0 resources. Both full-model runs used 128 tokens; all logits were finite.

| Model | Logits shape | Native AMX calls | First-token argmax |
|---|---|---:|---:|
| Qwen3-1.7B | `[1, 151936]` | 196 | 1813 |
| Qwen3.5-4B | `[1, 248320]` | 248 | 306 |

The dense/hybrid migration checks also passed against the whole-model HIR
evaluator, with maximum absolute errors of 0 and 5.74e-5 respectively. These
checks establish execution and small-model numerical agreement; the fixed-token
full-model smoke does not establish generation quality or performance.
