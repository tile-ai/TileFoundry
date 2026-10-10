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
# selection requested=compute-cost,memory,performance executed=compute-cost,memory,performance
# compute-cost flops=bf16:21476933632@logical,2834955239424@total,21476933632@cta,21476933632@thread;f32:167772160@logical,22145925120@total,167772160@cta,167772160@thread other-ops=integer:65@logical,16896@total,128@cta,128@thread precision=upper_bound
# memory traffic=gmem:r86.50MB/w280.00MB@logical,r31.45GB/w36.09GB@total,r244.00MB/w280.00MB@cta,r244.00MB/w280.00MB@thread;rmem:r1.26GB/w1.25GB@logical,r166.57GB/w166.55GB@total,r1.26GB/w1.26GB@cta,r1.26GB/w1.26GB@thread;smem:r240.00MB/w82.50MB@logical,r30.94GB/w30.94GB@total,r240.00MB/w240.00MB@cta,r240.00MB/w240.00MB@thread footprint=a:32.00KB;b:2.12MB;v20:36:128.00KB;v21:37:8.25MB footprint-precision=exact peak=gmem:522.00MB;rmem:128.00KB;smem:48.00KB persistent=gmem:250.00MB
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
grep -E 'tf.matmul|candidate   T.cuda.sm90.Wgmma|n=256, dtype=bf16, form=SS|tf.binary|tf.cast.*x=.*rmem|default' candidates.txt
```

```text
  v10:31  tf.matmul  per cta  lhs=Tensor[(128, 64), "bf16", "smem"]  rhs=Tensor[(64, 256), "bf16", "smem"]
    candidate   T.cuda.sm90.Wgmma  needs thread p0:p0+256, p0 % 128 = 0
                  n=256, dtype=bf16, form=SS
  v12:32  tf.binary  per cta  lhs=Tensor[(128, 256), "f32", "rmem"]  rhs=Tensor[(128, 256), "f32", "rmem"]  result=Tensor[(128, 256), "f32", "rmem"]
    default     T.binary
  v13:33  tf.cast  per cta  x=Tensor[(128, 256), "f32", "rmem"]  result=Tensor[(128, 256), "bf16", "rmem"]
    default     T.cast
```

The matmul still needs an atom and repetition choice. `T.binary` and the rmem cast have
one registered operation whose required attributes come from the HIR, so they say
`default`. Next inspect the exact WGMMA contract admitted by H200.

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
         ShardLayout(Layout((8, 2, 4, 2, 4, c), (1, 8, 16, 64, 128, 512)), (S(2), S(0), S(4)), Mesh(('thread',), ComposedLayout(None, p0, Layout(((4, 8, 4),), ((32, 4, 1),)))))
```

## 5. Write the scheduled HIR

The complete program below makes the structural decisions explicit:

- `Topology("cta", 132)` plus `g / bn / mi` states persistent G=16 traversal.
- `buffers=3` on the two `T.copy_async_tensor` loads states the BK64 TMA ring.
- `threads[0, :32]` is the producer; `threads[1:3, :]` are two consumers.
- `Wgmma(n=256, dtype=bf16, form=SS, a_major=K)` with `repeat=(2, 1, 4)` states eight K16 atoms.
- `T.copy(smem_layout=OUT_SMEM)` keeps the 64 KiB output staging tile outside all six
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
        a_smem = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))))
        b_smem = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))))
        out_smem = ComposedLayout(Swizzle(3, 4, 3), 0, Layout((128, (4, 64)), (64, (8192, 1))))
        with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for g in range(GM // GROUP_M):
                    for bn in range(GN):
                        start = (cta.persistent - (g * GN + bn) * GROUP_M) % CTAS
                        for mi in range(start, GROUP_M, CTAS):
                            m = (g * GROUP_M + mi) * BM
                            n = bn * BN
                            with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                                acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])

                            for k in tf.tile(K, BK):
                                with threads[0, :32] as _loader:
                                    lhs = tf.schedule(
                                        (a[m:m + BM, k],),
                                        op=T.copy_async_tensor(smem_layout=a_smem),
                                        buffers=STAGES,
                                    )
                                    rhs = tf.schedule(
                                        (b[k, n:n + BN],),
                                        op=T.copy_async_tensor(smem_layout=b_smem),
                                        buffers=STAGES,
                                    )

                                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                                    acc = tf.schedule(
                                        (acc, lhs, rhs),
                                        op=T.tiled_mma(atom=wgmma),
                                        repeat=(2, 1, 4),
                                    )

                            with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                                tile_out = tf.cast(acc, dtype="bf16")
                                staged = tf.schedule(
                                    (tile_out,), op=T.copy(smem_layout=out_smem)
                                )

                            with threads[0, :32] as _storer:
                                tile = tf.schedule((staged,), op=T.copy_async_tensor())
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
# memory traffic=gmem:r192.00MB/w306.00MB@logical,r133.68GB/w39.45GB@total,r1.01GB/w306.00MB@cta,r1.01GB/w306.00MB@thread;rmem:r2.71GB/w2.67GB@logical,r357.29GB/w357.20GB@total,r2.71GB/w2.71GB@cta,r11.51MB/w10.82MB@thread;smem:r1.01GB/w192.00MB@logical,r133.68GB/w133.68GB@total,r1.01GB/w1.01GB@cta,r192.31MB/w1020.07MB@thread footprint=a:256.00KB;b:288.00KB;v27:83:128.00KB;v28:84:8.25MB footprint-precision=exact peak=gmem:522.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:250.00MB

from __future__ import annotations

from tilefoundry.dsl import *  # noqa: F401, F403
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 132),), Layout((132,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        T.fill(out, 0.0)
        with Mesh(
            (Topology("thread", 384),), Layout((3, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            for g in range(0, 4, 1):
                for bn in range(0, 68, 1):
                    staged = T.tensor_view(
                        147456,
                        dtype='bf16',
                        storage="smem",
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    for mi in range(((cta.d0 - (32 * g)) - (16 * bn)) - (132 * (((cta.d0 - (32 * g)) - (16 * bn)) // 132)), 16, 132):
                        with Mesh(
                            scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as threads:
                            T.fill(acc, 0.0)
                        lhs_stages = (T.tensor_view(98304, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)), T.tensor_view(114688, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)), T.tensor_view(131072, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)))
                        rhs_stages = (T.tensor_view(0, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)), T.tensor_view(32768, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)), T.tensor_view(65536, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)))
                        for k in range(0, 5120, 64):
                            with scope[:1, :32] as scope_1:
                                tile = T.tensor_view(
                                    T.ptr_of(a[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 128, k:k + 64]),
                                    layout=Layout((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                                    T.copy_async_tensor(tile, lhs_stages[(k // 64) % 3])
                                tile_1 = T.tensor_view(
                                    T.ptr_of(b[k:k + 64, bn * 256:bn * 256 + 256]),
                                    layout=Layout((64, 256), (17408, 1)),
                                    shape=(64, 256),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                                    T.copy_async_tensor(tile_1, rhs_stages[(k // 64) % 3])
                            with Mesh(
                                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_2:
                                with Mesh(
                                    scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                                ) as threads_3:
                                    for o_m in range(0, 64, 64):
                                        for o_n in range(0, 256, 256):
                                            for o_k in range(0, 64, 64):
                                                acc_view = T.tensor_view(
                                                    T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                    layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                    shape=(64, 256),
                                                )
                                                lhs_view = T.tensor_view(
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k:o_k + 16]),
                                                    layout=ShardLayout(
                                                        layout=ComposedLayout(
                                                            inner=Swizzle(3, 4, 3),
                                                            offset=0,
                                                            outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                        ),
                                                        attrs=(B(), B(), B()),
                                                        mesh=threads_3,
                                                    ),
                                                    shape=(64, 16),
                                                )
                                                rhs_view = T.tensor_view(
                                                    T.ptr_of(rhs_stages[(k // 64) % 3][o_k:o_k + 16, o_n:o_n + 256]),
                                                    layout=ShardLayout(
                                                        layout=ComposedLayout(
                                                            inner=Swizzle(3, 4, 3),
                                                            offset=0,
                                                            outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                        ),
                                                        attrs=(B(), B(), B()),
                                                        mesh=threads_3,
                                                    ),
                                                    shape=(16, 256),
                                                )
                                                T.tiled_mma(
                                                    acc_view,
                                                    lhs_view,
                                                    rhs_view,
                                                    atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                                )
                                                acc_view_1 = T.tensor_view(
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
