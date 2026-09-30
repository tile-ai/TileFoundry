# Scheduling HIR into a kernel

`analyze` compares structures before a kernel exists. `schedule candidates` and
`schedule facts` then show which instructions can implement that structure, and
`tf.schedule` records the choices that `schedule finalize` lowers to TIR. This page
follows the 8192 x 5120 by 5120 x 17408 GEMM used to recover AtomSched's best H200
kernel.

To run this installed page, extract its two plain-HIR programs into files:

```bash
set -euo pipefail
for name in grid.py persistent.py; do
  awk -v tag="<!-- tilefoundry-source: $name -->" '
    $0 == tag { block=1; next }
    block && /^```python$/ { in_python=1; next }
    in_python && /^```$/ { in_python=0; block=0; next }
    in_python { print }
  ' schedule.md > "$name"
done
```

## 1. Start with one CTA per output tile

The first HIR is deliberately unscheduled. A 64 x 68 CTA mesh covers the output
with 128 x 256 tiles and streams K in blocks of 64. It says what each CTA computes,
but names no TIR instruction.

<!-- tilefoundry-source: grid.py -->

```python
#!/usr/bin/env python3
"""One 128 by 256 output tile per CTA on a 64 by 68 grid."""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
GM = M // BM
GN = N // BN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", GM * GN), Topology("thread", 384)))
class GRID:
    @func
    def gemm(a: Tensor[(M, K), "bf16"],
             b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GM, GN), names=("bm", "bn")) as cta:
            out = tf.zeros(Tensor[(M, N), "bf16"])
            m = cta.bm * BM
            n = cta.bn * BN
            acc = tf.zeros(Tensor[(BM, BN), "f32", "rmem"])
            for k in tile(K, BK):
                at = tf.reshard(a[m:m + BM, k], (BM, BK), "smem")
                bt = tf.reshard(b[k, n:n + BN], (BK, BN), "smem")
                part = tf.matmul(at, bt)
                acc = acc + tf.reshard(tf.cast(part, "f32"), (BM, BN), "rmem")
            tile_out = tf.reshard(tf.cast(acc, "bf16"), (BM, BN), "gmem")
            return tf.insert_slice(out, tile_out, (m, n))
```

```bash
set -euo pipefail
tilefoundry analyze grid.py grid.txt --performance
grep '^# performance root=' grid.txt
```

```text
# performance root=GRID::gemm predicted-ns=264035772 waves=33
```

The target can keep 132 CTAs resident, so 4352 CTAs require 33 waves. That is a
scheduling fact, not a reason to change the GEMM's arithmetic.

## 2. Make those CTAs persistent

The second HIR launches 132 CTAs and assigns tiles with two loops. For each M tile,
CTA `p` starts at `(p - mt * GN) % 132` in N and strides by 132. Across all CTAs
this is the same 4352-tile GEMM, covered exactly once.

<!-- tilefoundry-source: persistent.py -->

```python
#!/usr/bin/env python3
"""The same GEMM assigned to 132 persistent CTAs."""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
GM = M // BM
GN = N // BN
CTAS = 132


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", CTAS), Topology("thread", 384)))
class PERSISTENT:
    @func
    def gemm(a: Tensor[(M, K), "bf16"],
             b: Tensor[(K, N), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("p",)) as cta:
            out = tf.zeros(Tensor[(M, N), "bf16"])
            for mt in range(GM):
                for nt in range((cta.p - mt * GN) % CTAS, GN, CTAS):
                    m = mt * BM
                    n = nt * BN
                    acc = tf.zeros(Tensor[(BM, BN), "f32", "rmem"])
                    for k in tile(K, BK):
                        at = tf.reshard(a[m:m + BM, k], (BM, BK), "smem")
                        bt = tf.reshard(b[k, n:n + BN], (BK, BN), "smem")
                        part = tf.matmul(at, bt)
                        acc = acc + tf.reshard(tf.cast(part, "f32"), (BM, BN), "rmem")
                    tile_out = tf.reshard(tf.cast(acc, "bf16"), (BM, BN), "gmem")
                    out = tf.insert_slice(out, tile_out, (m, n))
            return out
```

```bash
set -euo pipefail
tilefoundry analyze persistent.py persistent.txt --performance
grep '^# performance root=' persistent.txt
```

```text
# performance root=PERSISTENT::gemm predicted-ns=13392246 waves=1
```

One wave replaces 33, and the model's prediction falls with it. These `predicted-ns`
values compare two HIR structures under the same static model; they are not wall-clock
times. Measure the generated kernel before making a performance claim.

## 3. Group 16 M tiles

Persistent execution lets tile order become an authored choice. Split the M index into
a group and an index inside that group, with N between them:

```text
for g in range(64 // 16):
    for bn in range(68):
        for mi in range((cta.p - (g * 68 + bn) * 16) % 132, 16, 132):
            m, n = (g * 16 + mi) * 128, bn * 256
```

This is the G=16 order: nearby work reuses B through L2, while 132 persistent CTAs
avoid relaunching 33 waves. AtomSched found the order under sustained load, where GPU
clock behavior also mattered; neither fact is encoded by `predicted-ns`. The maintained
fixture calls this same CTA axis `persistent` instead of the shorter `p`.

## 4. Ask what can implement the HIR

`candidates` works at each unscheduled site. The report distinguishes choices from
single registered operations that lowering will select by default.

```bash
set -euo pipefail
tilefoundry schedule candidates grid.py candidates.txt
grep -E 'tf.matmul|candidate   T.cuda.sm90.Wgmma|n=256, form=SS|tf.binary|tf.cast.*x=.*rmem|default' candidates.txt
```

```text
  v10:33  tf.matmul  per cta  lhs=Tensor[(128, 64), "bf16", "smem"]  rhs=Tensor[(64, 256), "bf16", "smem"]
    candidate   T.cuda.sm90.Wgmma  needs thread p0:p0+256, p0 % 128 = 0
                  n=256, form=SS
  v13:34  tf.binary  per cta  lhs=Tensor[(128, 256), "f32", "rmem"]  rhs=Tensor[(128, 256), "f32", "rmem"]  result=Tensor[(128, 256), "f32", "rmem"]
    default     T.binary
  v14:35  tf.cast  per cta  x=Tensor[(128, 256), "f32", "rmem"]  result=Tensor[(128, 256), "bf16", "rmem"]
    default     T.cast
```

The matmul still needs an atom and repetition choice. `T.binary` and the rmem cast have
one registered operation whose required attributes come from the HIR, so they say
`default`. Next inspect the exact WGMMA contract admitted by H200.

```bash
set -euo pipefail
tilefoundry schedule facts T.cuda.sm90.Wgmma --target nvidia.h200_sxm Wgmma.facts.txt
sed -n '1,21p' Wgmma.facts.txt
```

```text
T.cuda.sm90.Wgmma
  target             nvidia.h200_sxm
  target capability  wgmma.mma_async
  execution mesh     Mesh(('thread',), ComposedLayout(None, p0, Layout(((4, 8, 4),), ((32, 4, 1),))))
                       predicates:
                         each top-level mode has no backward step
                         each top-level mode reaches each of its own slots exactly once
                         p0 % 128 == 0
  parameters
    n        any value
               predicates:
                 8 <= n <= 256
                 n % 8 == 0
    form     any value
               predicates:
                 form in {Form.SS, Form.RS}
    a_major  form=SS  a_major in {Major.MN, Major.K} (default Major.MN)
             form=RS  Major.K
  operands
    C  shape=(64, n) dtype=f32 storage=rmem, held in 1 arrangement:
         ShardLayout(Layout((8, 2, 4, 2, 4, c), (1, 8, 16, 64, 128, 512)), (S(2), S(0), S(4)), Mesh(('thread',), ComposedLayout(None, p0, Layout(((4, 8, 4),), ((32, 4, 1),)))))
```

## 5. Write the scheduled HIR

The maintained example is
`tests/fixtures/schedule/hir/gemm_8192x17408x5120_optimal.py`; this page references it
instead of copying a second version that could drift. Its decisions map directly to the
CUDA kernel:

- `Topology("cta", 132)` plus `g / bn / mi` states persistent G=16 traversal.
- `buffers=3` on the two `T.copy_async_tensor` loads states the BK64 TMA ring.
- `threads[0, :32]` is the producer; `threads[1:3, :]` are two consumers.
- `Wgmma(n=256, form=SS, a_major=K)` with `repeat=(2, 1, 4)` states eight K16 atoms.
- `T.copy(smem_layout=OUT_SMEM)` keeps the 64 KiB output staging tile outside all six
  load-ring slots, then `T.copy_async_tensor` writes it to global memory.

The adjacent checklist
`tests/fixtures/schedule/gemm_8192x17408x5120_optimal.checklist.md` maps every
structure to both authored HIR and finalized TIR.

## 6. Finalize HIR into TIR

From a TileFoundry checkout, lower that maintained fixture with:

```text
tilefoundry schedule finalize \
  tests/fixtures/schedule/hir/gemm_8192x17408x5120_optimal.py \
  /tmp/gemm_8192x17408x5120_optimal.py
```

The command writes canonical Python TIR and prints nothing on success. The checked-in
golden is `tests/fixtures/schedule/tir/gemm_8192x17408x5120_optimal.py`: it retains the
three persistent loops, eight WGMMA atoms, three A and three B stage slots, and a
non-overlapping output staging interval.

## 7. Hand the TIR to the CUDA-writing agent

TIR settles value ownership, layouts, tile order, instruction atoms, and allocation.
The CUDA-writing agent preserves those decisions while adding the hardware protocol:
tensor maps and shared-memory descriptors, full/empty mbarriers and phases, WGMMA
fence/commit/wait, register fences and `setmaxnreg`, and the proxy fence before TMA
stores. Those are instruction encoding and asynchronous synchronization, not reasons
to silently change the schedule. Then `check` establishes agreement and measurement
decides whether the implementation is fast.
