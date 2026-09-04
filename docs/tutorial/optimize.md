# Making one step fast

`analyze` says what a program costs before any kernel exists, so a placement decision can
be priced before anyone writes a kernel for it. This page takes the step from
`tilefoundry tutorial migrate` and makes it fast, measuring after each change instead of
guessing which change was the good one.

Every `analyze` number below is static analysis of the authored program against the
rates the target publishes -- no kernel has to exist for it. The one kernel this page does
write is the Triton twin at the end, and what it is held to is agreement, not a number.

To run this installed page, extract its programs into files:

```bash
set -euo pipefail
for name in rms_norm_quant.py twin.py; do
  awk -v tag="<!-- tilefoundry-source: $name -->" '
    $0 == tag { block=1; next }
    block && /^```python$/ { in_python=1; next }
    in_python && /^```$/ { in_python=0; block=0; next }
    in_python { print }
  ' optimize.md > "$name"
done
```

## 1. Where the middle result lives

`Naive` splits the work at a `@func` boundary: `norm` produces the normalized rows and
`rms_norm_quant` quantizes them. A `@func` boundary is a real handover, so those rows are
written where the next function can read them.

<!-- tilefoundry-source: rms_norm_quant.py -->

```python
#!/usr/bin/env python3
"""The step from the migrate page, placed two ways: with a boundary, and without."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, ReduceKind, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

ROWS = 2
H = 7168  # this example's row width, not a field of any model
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
EPS = 1e-6  # config.json: rms_norm_eps

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", 1)
_THREAD = Topology("thread", ROWS * 4 * 32)


@module(entry="rms_norm_quant", target=_H200, topologies=(_CTA, _THREAD))
class Naive:
    """Normalize, hand the rows on, quantize them: two funcs, one boundary."""

    @func
    def norm(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (ROWS, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(a, (ROWS @ m.x, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                scaling = tf.reshard(gamma, (1, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                rows = tf.cast(held, "f32")
                mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
                normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * scaling
                return tf.reshard(normed, (ROWS, H), "gmem")

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        normed = norm(a, gamma)  # noqa: F821
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (ROWS, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(normed, (ROWS @ m.x, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                blocks = tf.reshape(tf.cast(held, "f32"), (ROWS, BLOCKS, BLOCK))
                scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
                quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
                return (
                    tf.reshard(tf.reshape(quant, (ROWS, H)), (ROWS, H), "gmem"),
                    tf.reshard(tf.reshape(scale, (ROWS, BLOCKS)), (ROWS, BLOCKS), "gmem"),
                )
```

```bash
set -euo pipefail
tilefoundry analyze rms_norm_quant.py:Naive Naive.txt --compute-cost --memory --roofline
grep -E '^# (traffic|roofline) ' Naive.txt
```

```text
# traffic traffic=gmem:r71680/w43456@r71680/w43456,rmem:r603496/w488344@r603496/w488344
# roofline ideal-ns=24 bound-by=memory
```

`bound-by=memory` says this step waits on memory, not on arithmetic: cutting flops
would buy nothing here, and cutting traffic is the change worth making. The rows crossing
the boundary are the traffic to cut -- 2 x 7168 bf16 is 28672 B, written once and read
back once.

`Fused` is the same arithmetic in one `@func`, so the normalized rows never leave a
register.

<!-- tilefoundry-source: rms_norm_quant.py -->

```python
@module(entry="rms_norm_quant", target=_H200, topologies=(_CTA, _THREAD))
class Fused:
    """One func: the normalized rows never leave a register."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (ROWS, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(a, (ROWS @ m.x, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                scaling = tf.reshard(gamma, (1, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                rows = tf.cast(held, "f32")
                mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
                normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * scaling
                blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
                scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
                quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
                return (
                    tf.reshard(tf.reshape(quant, (ROWS, H)), (ROWS, H), "gmem"),
                    tf.reshard(tf.reshape(scale, (ROWS, BLOCKS)), (ROWS, BLOCKS), "gmem"),
                )
```

```bash
set -euo pipefail
tilefoundry analyze rms_norm_quant.py:Fused Fused.txt --compute-cost --memory --roofline
grep -E '^# (traffic|roofline) ' Fused.txt
```

```text
# traffic traffic=gmem:r43008/w14784@r43008/w14784,rmem:r574824/w459672@r574824/w459672
# roofline ideal-ns=13 bound-by=memory
```

Reads fall by 28672 B and writes by 28672 B, exactly the rows that no longer make the
round trip, and `ideal-ns` falls with them. `rmem` traffic is unchanged: the same values
are still computed, they are just not spilled on the way.

## 2. A twin, and whether it still agrees

`analyze` prices a program; it says nothing about whether an implementation of it is
correct. The twin below is that implementation -- one Triton kernel, one row per program,
the whole step fused into it -- and `check` holds it to the authored program run by the
interpreter: with no `--expected`, the authored HIR *is* the reference.

Every output needs at least one predicate. The quantized tensor is discrete, so one wrong
value is a total failure and `equal` is the honest test; the scale is f32 and takes a
tolerance. `equal` on fp8 is a demanding thing to ask of a hand-written kernel -- a
different reduction order in the row sum would move the normalized values by one bf16 step
and some of them across an fp8 code. It passes here, and that is the claim the page is
making.

<!-- tilefoundry-source: twin.py -->

```python
#!/usr/bin/env python3
"""The runtime twin of the fused step, as one Triton kernel."""

import torch
import triton
import triton.language as tl
from rms_norm_quant import BLOCK, BLOCKS, EPS, FP8_MAX, H, ROWS, Fused

from tilefoundry.runtime import runtime_func, runtime_module

_TILE = triton.next_power_of_2(BLOCKS)


@triton.jit
def _rms_norm_quant(
    a_ptr, gamma_ptr, q_ptr, s_ptr, eps, fp8_max,
    H: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr, TILE: tl.constexpr,
):
    """One row per program: normalize it, then quantize it block by block."""
    row = tl.program_id(0)
    blocks = tl.arange(0, TILE)[:, None]
    lanes = tl.arange(0, BLOCK)[None, :]
    offsets = blocks * BLOCK + lanes
    live = blocks < BLOCKS

    a = tl.load(a_ptr + row * H + offsets, mask=live, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offsets, mask=live, other=0.0)
    normed = (a * tl.rsqrt(tl.sum(a * a) / H + eps)).to(tl.bfloat16) * gamma

    held = normed.to(tl.float32)
    scale = tl.max(tl.abs(held), axis=1, keep_dims=True) * (1.0 / fp8_max)
    quant = tl.clamp(held / scale, -fp8_max, fp8_max).to(tl.float8e4nv)

    tl.store(q_ptr + row * H + offsets, quant, mask=live)
    tl.store(s_ptr + row * BLOCKS + blocks, scale, mask=live)


@runtime_module(Fused)
class Fast:
    @runtime_func
    def rms_norm_quant(self, a, gamma):
        quant = torch.empty((ROWS, H), dtype=torch.float8_e4m3fn, device=a.device)
        scale = torch.empty((ROWS, BLOCKS), dtype=torch.float32, device=a.device)
        _rms_norm_quant[(ROWS,)](
            a, gamma, quant, scale, EPS, FP8_MAX,
            H=H, BLOCK=BLOCK, BLOCKS=BLOCKS, TILE=_TILE,
        )
        return quant, scale
```

```bash
set -euo pipefail
tilefoundry check twin.py:Fast --inputs random --weights random \
  --out 'output[0]' --fn equal \
  --out 'output[1]' --fn allclose --atol 1e-6 --rtol 1e-6
```

```text
twin.py:Fast
  reference: evaluator on Fused.rms_norm_quant
  inputs:    random (seed 0); activations actual bf16 (declared bf16)

  output[0]   fp8e4m3[2,7168]   ref_norm 12548.9
    equal                              mismatched 0 elements 14336 PASS
  output[1]   f32[2,56]   ref_norm 0.156977
    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS

PASS
```

`tilefoundry tutorial showcase` takes one kernel through six placement decisions, with the analysis output after each. `tilefoundry analyze --help` lists the other selectors, `tilefoundry check --help` states every predicate and how to derive a tolerance for a dtype, and `tilefoundry spec` is the normative reference behind both.
