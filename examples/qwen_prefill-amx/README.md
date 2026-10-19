# Qwen text prefill on Apple AMX

Qwen3-1.7B and Qwen3.5-4B text prefill, batch one, on Apple Silicon. The HIR
expresses the whole model; the scheduled HIR assigns its projections to AMX.
A Python runtime module implements the same module tree and calls C++ AMX
kernels through TVM FFI. Other operations run on the CPU using Torch.

## Files

| File | Purpose |
|---|---|
| `model.py` | Whole-model HIR, shared layer definitions and weight converters |
| `scheduled.py` | AMX projection schedule, packed layouts and worker assignment |
| `runtime_model.py` | Explicit Python runtime modules for projections, attention, GDN and layers |
| `kernels/prefill.cpp`, `kernels/runtime.hpp` | Native projection kernel and FFI entry |
| `kernels/__init__.py` | Build and load the native kernel on first use |
| `weights.py` | Q8_0 GGUF reading and resource aliases |
| `config.py` | Architecture parameters for the two models |
| `run.py` | Weight preparation, loading and one prefill call |

## Run

Requires an Apple Silicon Mac, Xcode Command Line Tools, Python 3.12+, this
TileFoundry revision, Torch, `gguf`, `safetensors` and `apache-tvm-ffi`.

```sh
python examples/qwen_prefill-amx/run.py --model 1.7b --checkpoint /path/to/Qwen3-1.7B-Q8_0.gguf
python examples/qwen_prefill-amx/run.py --model 4b --checkpoint /path/to/Qwen3.5-4B-Q8_0.gguf
```

The first run prepares the GGUF weights under `prepared/<model>/` and compiles
the C++ kernel under `kernels/.build/`. Subsequent runs reuse both. Use
`--prepared /path/to/prepared-weights` to choose another resource directory.
`--sequence` selects a positive multiple of 32; the default is 128. The entry
runs a fixed token sequence and prints the logits shape and next-token ID.

The runtime module can also be called directly with your own token IDs:

```python
from config import config
from runtime_model import make_runtime, inputs
from tilefoundry.runtime import SafetensorsResource

c = config("1.7b")
runtime = make_runtime(c, sequence=128)()
runtime.load(SafetensorsResource("prepared/1.7b", device="cpu"))
logits = runtime(*inputs(c, 128, token_ids))  # token_ids: CPU int64 tensor [128]
```

Run this snippet from the example directory. `runtime.module` is the scheduled
HIR module; `model.build(c, 128)` constructs its unscheduled counterpart.

## Execution arrangement

Q8_0 weights are dequantized to FP16 and packed from `[N,K]` to `[N/B,K,B]`,
with B=64, or 32 for narrow projections. Eight logical workers divide N panels
cyclically. Each AMX unit accumulates a 32×32 FP32 tile from FP16 outer products;
activations are packed into contiguous 32-row panels. The final one-row vocabulary
projection uses Torch. GDN, normalization, rotary embeddings and attention also
execute in the Python runtime using Torch.

AMX instruction contracts and the native instruction wrappers are provided by
TileFoundry. The example supplies the model-specific schedule and hand-written
runtime. It covers text prefill through logits, with no decode, MTP or vision.
