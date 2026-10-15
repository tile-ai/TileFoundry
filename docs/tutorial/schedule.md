# Choosing schedules and instructions for a backend

`analyze` compares structures before a kernel exists. `schedule candidates` and
`schedule facts` then show which instructions can implement that structure, and
`tf.schedule` records the choices that `schedule finalize` lowers to TIR. This page
follows the 8192 x 5120 by 5120 x 17408 GEMM used to recover AtomSched's best H200
kernel.

To run this installed page, extract its programs into files:

```bash
set -euo pipefail
for name in grid.py persistent.py optimal.py; do
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

from tilefoundry.dsl import *
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
            for k in tf.tile(K, BK):
                at = tf.reshard(a[m:m + BM, k], (BM, BK), "smem")
                bt = tf.reshard(b[k, n:n + BN], (BK, BN), "smem")
                part = tf.matmul(at, bt, out_dtype="f32")
                acc = acc + tf.reshard(part, (BM, BN), "rmem")
            tile_out = tf.reshard(tf.cast(acc, "bf16"), (BM, BN), "gmem")
            return tf.insert_slice(out, tile_out, (m, n))
```

```bash
set -euo pipefail
tilefoundry analyze grid.py grid.txt --performance
grep '^# performance root=' grid.txt
```

```text
# performance root=GRID::gemm predicted-ns=264231132 waves=33
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

from tilefoundry.dsl import *
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
                    for k in tf.tile(K, BK):
                        at = tf.reshard(a[m:m + BM, k], (BM, BK), "smem")
                        bt = tf.reshard(b[k, n:n + BN], (BK, BN), "smem")
                        part = tf.matmul(at, bt, out_dtype="f32")
                        acc = acc + tf.reshard(part, (BM, BN), "rmem")
                    tile_out = tf.reshard(tf.cast(acc, "bf16"), (BM, BN), "gmem")
                    out = tf.insert_slice(out, tile_out, (m, n))
            return out
```

```bash
set -euo pipefail
tilefoundry analyze persistent.py persistent.txt --compute-cost --memory --performance
if grep -qE '^#.*(error|advisory)="l2 ' persistent.txt; then
  printf '%s\n' 'persistent cache residency exceeds L2 capacity' >&2
  exit 1
fi
sed -n '/^from __future__/q;p' persistent.txt
```

```text
# analysis target=nvidia.h200_sxm module=PERSISTENT function=gemm topology=cta wave=132/132
# selection requested=compute-cost,memory,performance executed=compute-cost,memory,roofline,performance
# compute-cost flops=bf16:21476933632@logical,2834955239424@total,21476933632@cta,21476933632@thread;f32:167772160@logical,22145925120@total,167772160@cta,167772160@thread other-ops=integer:65@logical,16896@total,128@cta,128@thread precision=upper_bound
# memory traffic=gmem:r86.50MB/w280.00MB@logical,r31.45GB/w36.09GB@total,r244.00MB/w280.00MB@cta,r244.00MB/w280.00MB@thread;rmem:r1.26GB/w1.25GB@logical,r166.57GB/w166.55GB@total,r1.26GB/w1.26GB@cta,r1.26GB/w1.26GB@thread;smem:r240.00MB/w82.50MB@logical,r30.94GB/w30.94GB@total,r240.00MB/w240.00MB@cta,r240.00MB/w240.00MB@thread footprint=v20:36:128.00KB;v21:37:8.25MB;a:32.00KB;b:2.12MB footprint-precision=exact peak=gmem:522.00MB;rmem:256.00KB;smem:48.00KB persistent=gmem:250.00MB
# performance root=PERSISTENT::gemm predicted-ns=18317269 waves=1
```

One wave replaces 33, and the model's prediction falls with it. These `predicted-ns`
values compare two HIR structures under the same static model; they are not wall-clock
times. Measure the generated kernel before making a performance claim.

The memory report has no L2 capacity error or advisory; the command checks that before
showing the report. A one-time fill of the output is not an L2 reuse window.

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
clock behavior also mattered; neither fact is encoded by `predicted-ns`. The scheduled
program below calls this same CTA axis `persistent` instead of the shorter `p`.

## 4. Ask what can implement the HIR

`candidates` works at each unscheduled site. The report distinguishes choices from
single registered operations that lowering will select by default.

```bash
set -euo pipefail
tilefoundry schedule candidates grid.py candidates.txt
sed -n '/tf.matmul/,+4p; /refused.*T.binary/,+1p; /refused.*T.cast/,+1p' candidates.txt
```

```text
    refused     T.binary
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout((*execution_shape,), (*execution_strides,))))
    refused     T.binary
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout((*execution_shape,), (*execution_strides,))))
  v10:31  tf.matmul  per cta  lhs=Tensor[(128, 64), "bf16", (128, 64), "smem"]  rhs=Tensor[(64, 256), "bf16", (64, 256), "smem"]  result=Tensor[(128, 256), "f32", (128, 256), "rmem"]
    refused     T.cuda.sm90.Wgmma  n=256, dtype=f16, form=SS, a_major=MN, b_major=MN
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout(((*g, 4, 8, 4),), ((*gs, 32, 4, 1),))))
    refused     T.cuda.sm80.Mma
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout(((*g, 4, 8),), ((*gs, 1, 4),))))
    refused     T.binary
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout((*execution_shape,), (*execution_strides,))))
    refused     T.cast  cast_dtype=bf16
                  mesh: Mesh((Topology("cta", 4352),), layout=(64, 68), names=("bm", "bn")) does not match Mesh(('thread',), ComposedLayout(None, p0, Layout((*execution_shape,), (*execution_strides,))))
```

This baseline has only a CTA mesh. Each instruction shown requires a thread
mesh, so the first matching branch refuses it, including `T.binary` and `T.cast`.
A single registered operation becomes `default` only when it matches.

After stating the thread scope, the unscheduled matmul also needs a C fragment
layout for its accumulator. Its inferred compact result does not declare that
fragment, so use an explicit accumulator and a hand-written schedule, as below.
Next inspect the exact WGMMA contract admitted by H200.

```bash
set -euo pipefail
tilefoundry schedule facts T.cuda.sm90.Wgmma --target nvidia.h200_sxm Wgmma.facts.txt
sed -n '1,30p' Wgmma.facts.txt
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
    dtype    any value
               predicates:
                 dtype in {bf16, f16, fp8e4m3}
    form     any value
               predicates:
                 form in {Form.SS, Form.RS}
    a_major  dtype=bf16, form=SS  a_major in {Major.MN, Major.K} (default Major.MN)
             dtype=bf16, form=RS  Major.K
             dtype=f16, form=SS   a_major in {Major.MN, Major.K}
             dtype=f16, form=RS   Major.K
             dtype=fp8e4m3        Major.K
    b_major  dtype=bf16     b_major in {Major.MN, Major.K} (default Major.MN)
             dtype=f16      b_major in {Major.MN, Major.K}
             dtype=fp8e4m3  Major.K
  operands
    C  shape=(64, n) dtype=f32 storage=rmem, held in 1 arrangement:
         ShardLayout(Layout((4, 2, 8, c, 4, 2), (16, 8, 1, 512, 128, 64)), (S(0), S(2), S(4)), Mesh(('thread',), ComposedLayout(None, p0, Layout(((4, 8, 4),), ((32, 4, 1),)))))
```

## 5. Write the scheduled HIR

The complete program below makes the structural decisions explicit:

- `Topology("cta", 132)` plus `g / bn / mi` states persistent G=16 traversal.
- `buffers=3` on the two `tf.empty` load destinations states the BK64 TMA ring.
- `threads[0, :32]` is the producer; `threads[1:3, :]` are two consumers.
- `Wgmma(n=256, dtype=bf16, form=SS, a_major=K)` with `repeat=(2, 1, 4)` states eight K16 atoms.
- `tf.empty(Tensor[..., OUT_SMEM, "smem"])` keeps the 64 KiB output staging tile outside all six
  load-ring slots, then `T.copy_async_tensor` writes it to global memory.

<!-- tilefoundry-source: optimal.py -->

```python
"""State the structure of AtomSched's best H200 GEMM kernel.

The 132 persistent CTAs traverse groups of 16 M tiles in N-major order. A
producer warp fills a three-stage SW128 TMA ring while two consumer warpgroups
issue SS WGMMA, then stage each result in separate SW128 memory for a TMA store.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
STAGES = 3
CTAS = 132
GM = M // BM
GN = N // BN
GROUP_M = 16


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 384)),
)
class GEMM_8192X17408X5120_OPTIMAL:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for task in range(cta.persistent, GM * GN, CTAS):
                    g = task // (GN * GROUP_M)
                    bn = (task // GROUP_M) % GN
                    mi = task % GROUP_M
                    m = (g * GROUP_M + mi) * BM
                    n = bn * BN
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 32, 4 @ _compute.lane4, 2,), (16384, 16, 8, 1, 512, 128, 64)), "rmem"])

                    for k in tf.tile(K, BK):
                        with threads[0, :32] as _loader:
                            lhs_ring = tf.empty(
                                Tensor[
                                    (128, 64), "bf16",
                                    Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    "smem",
                                ],
                                buffers=STAGES,
                            )
                            lhs = tf.schedule(
                                (a[m:m + BM, k], lhs_ring),
                                op=T.copy_async_tensor(),
                            )
                            rhs_ring = tf.empty(
                                Tensor[
                                    (64, 256), "bf16",
                                    Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    "smem",
                                ],
                                buffers=STAGES,
                            )
                            rhs = tf.schedule(
                                (b[k, n:n + BN], rhs_ring),
                                op=T.copy_async_tensor(),
                            )

                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                            acc = tf.schedule(
                                (acc, lhs, rhs),
                                op=T.tiled_mma(atom=wgmma),
                                repeat=(2, 1, 4),
                            )

                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        tile_out = tf.cast(acc, dtype="bf16")
                        staged_dst = tf.empty(
                            Tensor[
                                (128, 256), "bf16",
                                Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                "smem",
                            ],
                        )
                        staged = tf.schedule(
                            (tile_out, staged_dst), op=T.copy()
                        )

                    with threads[0, :32] as _storer:
                        tile_dst = tf.empty(Tensor[(128, 256), "bf16", (128, 256)])
                        tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (m, n))

                return out
```

## 6. Finalize HIR into TIR

Finalize the extracted program and inspect the beginning of its TIR, including the
analysis report, persistent loops, and buffer placements:

```bash
set -euo pipefail
tilefoundry schedule finalize optimal.py optimal_tir.py
sed -n '1,114p' optimal_tir.py
```

```text
# analysis target=nvidia.h200_sxm module=GEMM_8192X17408X5120_OPTIMAL function=gemm topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r125.81MB/w276.12MB@logical,r16.22GB/w35.59GB@total,r125.81MB/w276.12MB@cta,r125.81MB/w276.12MB@thread;rmem:r336.27MB/w332.19MB@logical,r43.35GB/w43.34GB@total,r336.27MB/w336.19MB@cta,r1.40MB/w1.31MB@thread;smem:r125.81MB/w125.81MB@logical,r16.22GB/w16.22GB@total,r125.81MB/w125.81MB@cta,r23.33MB/w123.76MB@thread footprint=a:1.00MB;b:288.00KB;v24:101:64.00KB;v26:103:36.00MB footprint-precision=upper_bound peak=gmem:794.06MB;rmem:128.00KB;smem:208.00KB persistent=gmem:250.00MB
#   buffer=a holds=37.34MB time=none space=cta.persistent reuse=131.00MB fits=yes precision=upper_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], scratch: Tensor[(32768,), "bf16"], scratch_1: Tensor[(142606336,), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for task in range(cta.d0, 4352, 132):
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            attrs=(S(0), S(1), S(3), S(5)),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                            ), names=("d0", "d1", "d2", "d3")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_1:
                    T.fill(acc, 0.0)
                for k in range(0, 5120, 64):
                    with scope[:1, :32] as scope_2:
                        tile = T.tensor_view(
                            T.ptr_of(a[(((task // (68 * 16)) * 16) + (task % 16)) * 128:(((task // (68 * 16)) * 16) + (task % 16)) * 128 + 128, k:k + 64]),
                            layout=((128, 64), (5120, 1)),
                            shape=(128, 64),
                        )
                        T.copy_async_tensor(
                            tile,
                            lhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3],
                        )
                        tile_1 = T.tensor_view(
                            T.ptr_of(b[k:k + 64, ((task // 16) % 68) * 256:((task // 16) % 68) * 256 + 256]),
                            layout=((64, 256), (17408, 1)),
                            shape=(64, 256),
                        )
                        T.copy_async_tensor(
                            tile_1,
                            rhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3],
                        )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        for o_m in range(0, 128, 128):
                            for o_n in range(0, 256, 256):
                                for o_k in range(0, 64, 64):
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k:o_k + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_k:o_k + 16, o_n:o_n + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(rhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(rhs_ring_stages[((((task - cta.d0) // 132) * 80) + (k // 64)) % 3][o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
```

The three `g / bn / mi` loops and `cta.d0` survive lowering. The three RHS slots
occupy [0, 98304), the three LHS slots occupy [98304, 147456), and output staging
occupies [147456, 212992). The report and views expose those disjoint intervals;
the TIR continues with the WGMMA issues and writeback.

The initial `T.fill(out, 0.0)` follows this page's `tf.zeros` plus `insert_slice`
representation of the output; a CUDA implementation need not clear the whole output
before overwriting every tile.

## 7. Implement the TIR for a concrete backend

TIR settles value ownership, layouts, tile order, instruction atoms, and allocation.
A backend implementation preserves those decisions while supplying instruction
encoding and synchronization.

For CUDA on H200, that protocol includes tensor maps and shared-memory descriptors,
full/empty mbarriers and phases, WGMMA fence/commit/wait, register fences and
`setmaxnreg`, and the proxy fence before TMA stores. These are not reasons to
silently change the schedule. Then `check` establishes agreement and measurement
decides whether the implementation is fast.
