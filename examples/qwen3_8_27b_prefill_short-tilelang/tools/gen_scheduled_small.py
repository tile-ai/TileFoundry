#!/usr/bin/env python3
"""Scheduled HIR + finalized TIR for the small kernels around the GEMMs.

    scheduled/small/<kernel>_T<T>.py  ->  tir/small/<kernel>_T<T>.py

  rmsnorm      model.py: tf.rms_norm(h, 1 + gamma) of every layer's two pre-norms
               (runtime/kernels/small.py rmsnorm_kernel): one warp per row, `rows`
               rows per CTA (plan.norm_rows), lane l owns the 4-element chunks
               l, l+32, ..., one cross-lane reduce (torch's summation order).
  rmsnorm_last final_norm of the last row only (the head reads only that row).
  reduce_sk    the split-K finish of o_proj/down at the lengths that split
               (gemm.py splitk_reduce_kernel): C = bf16(R + bf16(sum_s P[s])).
  lm_head      model.py lm_head: one row x (248320, 5120)^T, no MMA: 16 weight rows
               per CTA, 8 warps x 2 rows, each lane 8 contiguous elements per
               256-wide chunk, a cross-lane reduce (small.py gemv_kernel).
  embed        model.py embed: table rows gathered by token id (small.py embed_kernel).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: the `tilefoundry` CLI: $TILEFOUNDRY, or whatever is on PATH (this example is not
#: installed and does not carry a virtualenv of its own).
TILEFOUNDRY = os.environ.get("TILEFOUNDRY") or shutil.which("tilefoundry") or "tilefoundry"
sys.path.insert(0, str(ROOT))
from runtime import plan  # noqa: E402
from runtime.kernels.gemm import ba_split  # noqa: E402

H, V = 5120, 248320
PRE = '''from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

'''


def rmsnorm(M, name, doc, rows):
    return f'''"""Scheduled HIR: {doc}. Rows M = {M}.
Runtime kernel: runtime/kernels/small.py rmsnorm_kernel(M={M}, H=5120, rows={rows}).

Decomposition (as the kernel): CTA = {rows} row(s), one warp per row; lane l holds the
4-element chunks l, l + 32, ..., l + 32 * (NCH - 1) of its row (register index 4c + i);
the mean of squares is one cross-lane reduce. The kernel fixes the f32 summation order
to torch's (per-lane accumulators per chunk position, ((a0+a1)+a2)+a3, shfl_down
16..1, times f32(1/H)) so its output is bitwise equal to the HIR; this program states
the reduce without an order.
"""
{PRE}M = {M}
H = {H}
ROWS = {rows}
NCH = H // 128
EPS = 1e-6


@module(entry="rmsnorm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M // ROWS), Topology("thread", 32 * ROWS)))
class SCHED_{name.upper()}:
    @func
    def rmsnorm(x: Tensor[(M, H), "bf16"], gamma: Tensor[(H,), "bf16"]) -> Tensor[(M, H), "bf16"]:
        with Mesh(("cta",), layout=(M // ROWS,), names=("blk",)) as cta:
            out = tf.zeros(Tensor[(M, H), "bf16"])
            with Mesh(("thread",), layout=(ROWS, 32), names=("w", "lane")) as th:
                xr = tf.schedule((x[cta.blk * ROWS:cta.blk * ROWS + ROWS, :], tf.empty(Tensor[(ROWS, H), "bf16", ((ROWS @ th.w, NCH, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                gr = tf.schedule((tf.reshape(gamma, new_shape=(1, H)), tf.empty(Tensor[(1, H), "bf16", ((1, NCH, 32 @ th.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                xf = tf.cast(xr, dtype="f32")
                ms = tf.reduce(xf * xf, axes=(-1,), keepdim=True, kind="mean")
                y = tf.cast(xf * tf.rsqrt(ms + EPS) * (tf.cast(gr, dtype="f32") + 1.0), dtype="bf16")
                yo = tf.schedule((y, tf.empty(Tensor[(ROWS, H), "bf16", (ROWS, H)])), op=T.copy())
                out = tf.insert_slice(out, yo, (cta.blk * ROWS, 0))
            return out
'''


def reduce_sk(M, S):
    return f'''"""Scheduled HIR: split-K finish of o_proj/down at M = {M}, {S} slices: C = bf16(R + bf16(sum_s P[s])).
Runtime kernel: runtime/kernels/gemm.py splitk_reduce_kernel(M={M}, N=5120, S={S}, residual=True): 1024 threads x 8 elements per CTA."""
{PRE}M = {M}
N = 5120
S = {S}
BLOCK = 1024
PER = 8
SPAN = BLOCK * PER
TILES = M * N // SPAN


@module(entry="reduce_sk", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", TILES), Topology("thread", BLOCK)))
class SCHED_REDUCE_SK_T{M}:
    @func
    def reduce_sk(p: Tensor[(S, M * N), "f32"], r: Tensor[(M * N,), "bf16"]) -> Tensor[(M * N,), "bf16"]:
        with Mesh(("cta",), layout=(TILES,), names=("tile",)) as cta:
            out = tf.zeros(Tensor[(M * N,), "bf16"])
            base = cta.tile * SPAN
            with Mesh(("thread",), layout=(BLOCK,), names=("t",)) as th:
                pp = tf.schedule((p[:, base:base + SPAN], tf.empty(Tensor[(S, SPAN), "f32", ((S, BLOCK @ th.t, PER), (PER, 0, 1)), "rmem"])), op=T.copy())
                rr = tf.schedule((r[base:base + SPAN], tf.empty(Tensor[(SPAN,), "bf16", ((BLOCK @ th.t, PER), (0, 1)), "rmem"])), op=T.copy())
                total = tf.reduce(pp, axes=(0,), keepdim=False, kind="sum")
                c = tf.cast(tf.cast(rr, dtype="f32") + tf.cast(tf.cast(total, dtype="bf16"), dtype="f32"), dtype="bf16")
                co = tf.schedule((c, tf.empty(Tensor[(SPAN,), "bf16", (SPAN,)])), op=T.copy())
                out = tf.insert_slice(out, co, base)
            return out
'''


def reduce_ba(M, S):
    return f'''"""Scheduled HIR: finish of the GDN b|a projections at M = {M}, {S} K slices:
ba = bf16(sum_s P[s]) for the 96 columns [b (48) | a (48)].
Implements: model.py Qwen3_8DeltaNetLayer.gated_delta_net, matmul(x, w_b, NK) and
matmul(x, w_a, NK) (second half: the slices come from scheduled/gemm/gdn_ba_T{M}.py).
Runtime kernel: runtime/kernels/gemm.py ba_reduce_kernel(M={M}, S={S}, 16640, 16384):
one CTA per row, thread n owns column n.
Differences from the kernel: the kernel adds the slices from the last to the first
(cuBLAS's split-K order, see gemm.py BA_N); this program states the sum without an
order. The kernel writes the 96 values into columns 16384..16479 of the (M, 16640)
GDN projection buffer; this program's result is the (M, 96) block itself.
"""
{PRE}M = {M}
S = {S}
N = 96


@module(entry="reduce_ba", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M), Topology("thread", N)))
class SCHED_REDUCE_BA_T{M}:
    @func
    def reduce_ba(p: Tensor[(S, M, N), "f32"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(M,), names=("row",)) as cta:
            out = tf.zeros(Tensor[(M, N), "bf16"])
            with Mesh(("thread",), layout=(N,), names=("n",)) as th:
                pp = tf.schedule((tf.reshape(p[:, cta.row:cta.row + 1, :], new_shape=(S, N)), tf.empty(Tensor[(S, N), "f32", ((S, N @ th.n), (1, 0)), "rmem"])), op=T.copy())
                total = tf.reduce(pp, axes=(0,), keepdim=True, kind="sum")
                c = tf.cast(total, dtype="bf16")
                co = tf.schedule((c, tf.empty(Tensor[(1, N), "bf16", (1, N)])), op=T.copy())
                out = tf.insert_slice(out, co, (cta.row, 0))
            return out
'''


def lm_head():
    return f'''"""Scheduled HIR: lm_head over the last row, (1, 5120) x (248320, 5120)^T, as a GEMV (no MMA:
one row reads every weight byte once, 2.54 GB, memory-bound at every length).
Runtime kernel: runtime/kernels/small.py gemv_kernel(N=248320, K=5120, rows=16, threads=256)."""
{PRE}N = {V}
K = {H}
ROWS = 16
WARPS = 8
RPW = ROWS // WARPS
VEC = 8
CHUNKS = K // (32 * VEC)


@module(entry="lm_head", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", N // ROWS), Topology("thread", 256)))
class SCHED_LM_HEAD:
    @func
    def lm_head(x: Tensor[(1, K), "bf16"], w: Tensor[(N, K), "bf16"]) -> Tensor[(1, N), "bf16"]:
        with Mesh(("cta",), layout=(N // ROWS,), names=("blk",)) as cta:
            out = tf.zeros(Tensor[(1, N), "bf16"])
            n = cta.blk * ROWS
            with Mesh(("thread",), layout=(WARPS, 32), names=("warp", "lane")) as th:
                wt = tf.schedule((w[n:n + ROWS, :], tf.empty(Tensor[(ROWS, K), "bf16", ((WARPS @ th.warp, RPW, CHUNKS, 32 @ th.lane, VEC), (0, CHUNKS * VEC, VEC, 0, 1)), "rmem"])), op=T.copy())
                xt = tf.schedule((x, tf.empty(Tensor[(1, K), "bf16", ((1, CHUNKS, 32 @ th.lane, VEC), (0, VEC, 0, 1)), "rmem"])), op=T.copy())
                prod = tf.cast(wt, dtype="f32") * tf.cast(xt, dtype="f32")
                logit = tf.cast(tf.reduce(prod, axes=(-1,), keepdim=False, kind="sum"), dtype="bf16")
                lo = tf.schedule((tf.reshape(logit, new_shape=(1, ROWS)), tf.empty(Tensor[(1, ROWS), "bf16", (1, ROWS)])), op=T.copy())
                out = tf.insert_slice(out, lo, (0, n))
            return out
'''


def embed(M):
    return f'''"""Scheduled HIR: embedding gather for M = {M} prompt tokens: out[t] = table[ids[t]].
Runtime kernel: runtime/kernels/small.py embed_kernel(M={M}): one CTA per token, the row staged through shared memory by cp.async, then stored."""
{PRE}M = {M}
V = {V}
H = {H}
TH = 128
PER = H // TH


@module(entry="embed", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", M), Topology("thread", TH)))
class SCHED_EMBED_T{M}:
    @func
    def embed(ids: Tensor[(M,), "i64"], table: Tensor[(V, H), "bf16"]) -> Tensor[(M, H), "bf16"]:
        with Mesh(("cta",), layout=(M,), names=("row",)) as cta:
            out = tf.zeros(Tensor[(M, H), "bf16"])
            with Mesh(("thread",), layout=(TH,), names=("t",)) as th:
                one = ids[cta.row:cta.row + 1]
                row = tf.schedule((table, tf.empty(Tensor[(1, H), "bf16", (1, H), "smem"]), one), op=T.copy_async(fill=0.0))
                ro = tf.schedule((row, tf.empty(Tensor[(1, H), "bf16", (1, H)])), op=T.copy())
                out = tf.insert_slice(out, ro, (cta.row, 0))
            return out
'''


def main():
    sdir, tdir = ROOT / "scheduled/small", ROOT / "tir/small"
    sdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    jobs = {"rmsnorm_last": rmsnorm(1, "rmsnorm_last", "final_norm of the last row", 1), "lm_head": lm_head()}
    for T in (128, 256, 512, 1024):
        jobs[f"rmsnorm_T{T}"] = rmsnorm(T, f"rmsnorm_T{T}", "pre-norm of every layer (gamma_in, gamma_post)", plan.for_length(T).norm_rows)
        jobs[f"embed_T{T}"] = embed(T)
        p = plan.for_length(T)
        sks = {p.gemm[k].split_k for k in ("o_proj", "down") if p.gemm[k].split_k > 1}
        for S in sks:
            jobs[f"reduce_sk{S}_T{T}"] = reduce_sk(T, S)
        jobs[f"reduce_ba_T{T}"] = reduce_ba(T, ba_split(T))
    fails = 0
    for name, src in jobs.items():
        (sdir / f"{name}.py").write_text(src)
        r = subprocess.run([TILEFOUNDRY, "schedule", "finalize", str(sdir / f"{name}.py"),
                            str(tdir / f"{name}.py")], capture_output=True, text=True)
        fails += r.returncode != 0
        print(f"{name:20s} finalize {'OK' if r.returncode == 0 else 'FAIL'}"
              + ("" if r.returncode == 0 else "\n    " + r.stderr.strip()[:700].replace("\n", "\n    ")), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
