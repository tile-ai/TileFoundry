# Describing a published step

The first step of a migration has one criterion: the program you author must agree with
the published implementation. `check` is what says so, and it says it about one output at
a time -- so the answer is never "close enough", it is a predicate per output with a bound
you chose.

The step here is one sublayer of a production model: RMS-normalize a row, take a blockwise
absolute maximum, quantize to fp8. It returns two tensors, so `check` addresses them as
`output[0]` and `output[1]`.

To run this installed page, extract its program and fetch the published fields it is
measured against:

```bash
set -euo pipefail
awk -v tag="<!-- tilefoundry-source: rms_norm_quant.py -->" '
  $0 == tag { block=1; next }
  block && /^```python$/ { in_python=1; next }
  in_python && /^```$/ { in_python=0; block=0; next }
  in_python { print }
' migrate.md > rms_norm_quant.py
published=$(tilefoundry models deepseek_v4_flash --source 2>/dev/null | sed -n '1p')
cp "$published/config.json" .
```

### The published side

The reference is the real class, not a paraphrase of it: `transformers`'
`LlamaRMSNorm`, with the epsilon the model publishes, followed by the quantization the
model's own `quantization_config` describes. The seed is fixed, so every number this page
shows is reproducible.

```python
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRMSNorm

published = json.loads(Path("config.json").read_text(encoding="utf-8"))
eps = published["rms_norm_eps"]
block = published["quantization_config"]["weight_block_size"][1]
fmt = published["quantization_config"]["fmt"]
print(f"rms_norm_eps={eps}  weight_block_size={block}  fmt={fmt}")

ROWS, H, FP8_MAX = 2, 7168, 448.0
torch.manual_seed(0)
a = torch.randn(ROWS, H, dtype=torch.bfloat16)
gamma = (1.0 + 0.02 * torch.randn(H)).to(torch.bfloat16)

norm = LlamaRMSNorm(H, eps=eps)
norm.weight = torch.nn.Parameter(gamma.clone())
with torch.no_grad():
    normed = norm(a)

blocks = normed.float().reshape(ROWS, H // block, block)
scale = blocks.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)
quant = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

torch.save(a, "x.pt")
torch.save([quant.reshape(ROWS, H), scale.reshape(ROWS, H // block)], "expected.pt")
save_file({"gamma": gamma.reshape(1, H).contiguous()}, "model.safetensors")
print(f"wrote x.pt expected.pt model.safetensors for {tuple(a.shape)} {a.dtype}")
```

```text
rms_norm_eps=1e-06  weight_block_size=128  fmt=e4m3
wrote x.pt expected.pt model.safetensors for (2, 7168) torch.bfloat16
```

`rms_norm_eps`, the block width and the format are **published fields**. They are not
constants to remember, and the two that look like round numbers are the ones most often
remembered wrong.

One thing no field tells you: *where* the result lands in bf16. That is in the code, on the
last line of `LlamaRMSNorm.forward` -- `self.weight * hidden_states.to(input_dtype)`, which
casts first and scales second. The version below scales first.

<!-- tilefoundry-source: rms_norm_quant.py -->

```python
#!/usr/bin/env python3
"""One published step, authored as HIR: RMS norm, then blockwise FP8 quantization."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

ROWS = 2
H = 7168  # this example's row width, not a field of the model above
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
EPS = 1e-6  # config.json: rms_norm_eps


@module(
    entry="rms_norm_quant",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class RmsNormQuant:
    """The step as I remember it: scale by gamma, then land in bf16."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        rows = tf.cast(a, "f32")
        mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
        normed = tf.cast(rows * tf.rsqrt(mean + EPS) * tf.cast(gamma, "f32"), "bf16")
        blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
        scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
        quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
        return tf.reshape(quant, (ROWS, H)), tf.reshape(scale, (ROWS, BLOCKS))
```

```bash
set -euo pipefail
set +e
tilefoundry check rms_norm_quant.py:RmsNormQuant --inputs files:x.pt --weights ckpt:. \
  --expected expected.pt \
  --out 'output[0]' --fn equal \
  --out 'output[1]' --fn allclose --atol 1e-6 --rtol 1e-6
status=$?
set -e
[ "$status" -ne 0 ] || { echo "expected this version to be refused" >&2; exit 1; }
```

```text
rms_norm_quant.py:RmsNormQuant
  reference: expected.pt
  inputs:    files:x.pt; activations actual bf16 (declared none); files x.pt: 1 tensor(s) bf16[2, 7168]

  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 432 elements 14336 FAIL
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 6.87445e-05  FAIL

FAIL

  warning: FAIL says the candidate and reference differ, not which side is closer to
           truth. The reference may carry its own rounding; check compares only
           against it. Establishing accuracy needs an independent high-precision
           reference, which check does not run.
```

Two outputs, two answers, and they disagree by different amounts. The quantized tensor
loses most of the difference -- fp8e4m3 keeps three mantissa bits, so a step of about
4e-3 usually lands on the same code -- and 432 of 14336 elements survive it. The f32 scale
keeps all of it: `max_violation` is how far the worst element is *past* the bound, so
6.87e-5 against a bound of about 1e-6 says the disagreement is real and small.

`FAIL` says the two sides differ. It does not say which one is right; `check` prints that
warning itself, because the reference carries its own rounding. Here the published class is
right by definition -- it is what a user of this model runs.

The fix is one line: land in bf16, then scale. The block below is tagged as a second
source for the same file, so the extraction command after it writes over
`rms_norm_quant.py`, and the
`check` that follows is the *same command*, character for character, as the one that
failed.

<!-- tilefoundry-source: rms_norm_quant-fixed -->

```python
#!/usr/bin/env python3
"""One published step, authored as HIR: RMS norm, then blockwise FP8 quantization."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

ROWS = 2
H = 7168  # this example's row width, not a field of the model above
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
EPS = 1e-6  # config.json: rms_norm_eps


@module(
    entry="rms_norm_quant",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class RmsNormQuant:
    """The step as the published class writes it: land in bf16, then scale by gamma."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        rows = tf.cast(a, "f32")
        mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
        normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * gamma
        blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
        scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
        quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
        return tf.reshape(quant, (ROWS, H)), tf.reshape(scale, (ROWS, BLOCKS))
```

```bash
set -euo pipefail
awk -v tag="<!-- tilefoundry-source: rms_norm_quant-fixed -->" '
  $0 == tag { block=1; next }
  block && /^```python$/ { in_python=1; next }
  in_python && /^```$/ { in_python=0; block=0; next }
  in_python { print }
' migrate.md > rms_norm_quant.py
```

```bash
set -euo pipefail
tilefoundry check rms_norm_quant.py:RmsNormQuant --inputs files:x.pt --weights ckpt:. \
  --expected expected.pt \
  --out 'output[0]' --fn equal \
  --out 'output[1]' --fn allclose --atol 1e-6 --rtol 1e-6
```

```text
rms_norm_quant.py:RmsNormQuant
  reference: expected.pt
  inputs:    files:x.pt; activations actual bf16 (declared none); files x.pt: 1 tensor(s) bf16[2, 7168]

  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 0 elements 14336 PASS
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS

PASS
```

Once the program agrees, `tilefoundry tutorial optimize` is the second step: making it fast without losing the agreement. `tilefoundry check --help` states every predicate and the arithmetic for choosing a bound, and `tilefoundry spec` is the normative reference.
