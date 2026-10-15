#!/usr/bin/env python3
"""Generate the scheduled HIR of every GEMM the plan uses, and finalize each to TIR.

One scheduled program per (linear, prompt length): the tile shape, ring depth,
split and epilogue come from runtime/plan.py, the authored computation from
model.py (which function each linear belongs to is written in the header of the
generated file). Output:
    scheduled/gemm/<linear>_T<T>.py      scheduled HIR (a Module with one @func)
    tir/gemm/<linear>_T<T>.py            `tilefoundry schedule finalize` of it

usage: tools/gen_scheduled.py [--only NAME] [--lengths 128 256 ...]
"""
import argparse
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

H, I, CONV, HV, DV = 5120, 17408, 10240, 48, 128
GDN_IN_PAD = 16640
ATT_IN = 14336
# linear -> (N, K, epilogue, the authored function it implements)
LINEARS = {
    "gdn_in": (GDN_IN_PAD, H, "store",
               "Qwen3_8DeltaNetLayer.gated_delta_net: matmul(x, w_qkv|w_z|w_b|w_a, NK) (packed rows, padded to 16640)"),
    "att_in": (ATT_IN, H, "store",
               "Qwen3_8AttentionLayer.gated_attention: matmul(x, w_q|w_k|w_v, NK) (packed rows)"),
    "gate_up": (I, H, "swiglu",
                "Qwen3_8*Layer.swiglu: silu(matmul(x, w_gate, NK)) * matmul(x, w_up, NK)"),
    "o_proj": (H, HV * DV, "residual",
               "Qwen3_8*Layer.layer: hidden + matmul(gated, w_out|w_o, NK)"),
    "down": (H, I, "residual",
             "Qwen3_8*Layer.layer: attended + matmul(silu(gate)*up, w_down, NK)"),
    "gdn_ba": (96, H, "store",
               "Qwen3_8DeltaNetLayer.gated_delta_net: matmul(x, w_b, NK), matmul(x, w_a, NK) as [w_b|w_a] rows, "
               "K split like cuBLAS for N=48 (runtime/kernels/gemm.py BA_N); the f32 slices are finished by "
               "scheduled/small/reduce_ba_T<T>.py, which overwrites the b|a columns the gdn_in GEMM also wrote"),
}

HEADER = '''"""Scheduled HIR: {linear} at prompt length {T}.

Implements: {what}
Plan row: runtime/plan.py, length {T}: {cfg}
Finalized TIR: tir/gemm/{linear}_T{T}.py   Runtime kernel: runtime/kernels/gemm.py gemm_kernel(..., "{epi_rt}")

Structure: one CTA per ({BM} x {BN}) output tile{split_txt}, tiles launched M-fastest;
384 threads = warpgroup 0
(lanes 0..31 issue TMA) + warpgroups 1..2 (WGMMA m64n{NI}k16, warpgroup g owns rows
[64g, 64g+64)); A and W tiles stream through {S}-deep 128B-swizzled TMA rings; W is
the checkpoint's (out, in) weight, read K-major (b_major=K) through a bitcast view.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = {M}
N = {N}
K = {K}
BM = {BM}
BN = {BN}
BK = {BK}
STAGES = {S}
GM = M // BM
GN = N // BN
'''


def acc_layout(BN):
    return (f"((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, {BN // 8}), "
            f"({BN * 64}, 1, 8, 16, 64, 128, 512))")


def ring(rows, bk=64):
    """(rows, bk) K-major tile: 64-column 128B-swizzled panels; a 128-wide K tile is
    two panels, the second one rows*64 elements further (two TMA boxes)."""
    if bk == 64:
        return f"Layout((({rows // 8}, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)"
    return f"Layout((({rows // 8}, 8), (2, 4, 16)), ((512, 64), ({rows * 64}, 16, 1))) | Swizzle(3, 4, 3)"


def ring_t(rows, bk=64):
    if bk == 64:
        return f"Layout(((4, 16), ({rows // 8}, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3)"
    return f"Layout(((2, 4, 16), ({rows // 8}, 8)), (({rows * 64}, 16, 1), (512, 64))) | Swizzle(3, 4, 3)"


COMPUTE = 'Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4"))'


def mainloop(srcs, k_lo, k_hi, k_base=None):
    """srcs: [(weight param, ring name, accumulator)] sharing one A ring and one K loop.
    With k_base (a split-K slice start) the loop runs over [0, k_hi) and windows are
    offset by k_base."""
    if k_base is None:
        head = f"                for k in tf.tile({k_lo}, {k_hi}, BK):\n"
        win = "k"
    else:
        head = f"                for kk in range(0, {k_hi}, BK):\n"
        win = f"{k_base} + kk:{k_base} + kk + BK"
    loads = "".join(
        f'''                        {ring_name} = tf.empty(Tensor[(BN, BK), "bf16", {{RING}}, "smem"], buffers=STAGES)
                        rhs_{i} = tf.schedule(({src}[n:n + BN, {win}], {ring_name}), op=T.copy_async_tensor())
''' for i, (src, ring_name, _) in enumerate(srcs))
    mmas = "".join(
        f'''                        rhs_t{i} = tf.bitcast(rhs_{i}, {{RING_T}})
                        {acc} = tf.schedule(({acc}, lhs, rhs_t{i}), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, {{KREP}}))
''' for i, (_, _, acc) in enumerate(srcs))
    return (head + f'''                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", {{LHS}}, "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, {win}], lhs_ring), op=T.copy_async_tensor())
''' + loads + f"                    with {COMPUTE} as _compute:\n" + mmas)


def gemm_source(linear, T, cfg):
    N, K, epi, what = LINEARS[linear]
    BM, BN, BK, S, SK = cfg.BM, cfg.BN, cfg.BK, cfg.stages, cfg.split_k
    assert BM == 128 and BK in (64, 128) and T % BM == 0, (linear, T, cfg)
    pad_note = ""
    if N % BN:
        n_real = N
        N = -(-N // BN) * BN
        pad_note = (f" N is {n_real}; the program is stated over {N} = {N // BN} whole tiles: weight rows"
                    f" >= {n_real} are TMA out-of-bounds zero fill and output columns >= {n_real} are clipped by the"
                    " TMA store, so the valid region is the authored computation.")
    what = what + pad_note
    two = epi == "swiglu"
    NI = BN
    split_txt = f", K split in {SK} slices (f32 partial sums, then a reduce + residual pass)" if SK > 1 else ""
    epi_rt = {"store": "splitk" if SK > 1 else "store", "residual": "splitk" if SK > 1 else "residual", "swiglu": "swiglu2"}[epi]
    src = HEADER.format(linear=linear, T=T, what=what, cfg=cfg, epi_rt=epi_rt, BM=BM, BN=BN, split_txt=split_txt,
                        NI=NI, S=S, M=T, N=N, K=K, BK=BK)
    if SK > 1:
        src += f"SPLIT = {SK}\nKS = K // SPLIT\n"
        topo = "GM * GN * SPLIT"
    else:
        topo = "GM * GN"
    cls = f"SCHED_{linear.upper()}_T{T}"
    params = 'a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"]'
    if two:
        params = 'a: Tensor[(M, K), "bf16"], w_gate: Tensor[(N, K), "bf16"], w_up: Tensor[(N, K), "bf16"]'
    if epi == "residual" and SK == 1:
        params += ', residual: Tensor[(M, N), "bf16"]'
    ret = 'Tensor[(SPLIT, M, N), "f32"]' if SK > 1 else 'Tensor[(M, N), "bf16"]'
    # Tile order: the runtime kernel launches T.use_swizzle(panel=8) when cfg.swizzle,
    # i.e. tilelang's rasterization2DRow<8>; with GM = T/128 <= 8 tiles along M that is
    # one panel, which maps launch index p to (m = p % GM, n = p // GM): M fastest.
    # A (GN, GM) mesh named (bn, bm) states exactly that order (last axis fastest).
    assert T // BM <= 8
    if SK > 1 and linear == "gdn_ba":   # ba_splitk_kernel: grid (GN, GM, SPLIT), N tiles fastest
        mesh = 'Mesh(("cta",), layout=(SPLIT, GM, GN), names=("ks", "bm", "bn"))'
    elif SK > 1:
        mesh = 'Mesh(("cta",), layout=(SPLIT, GN, GM), names=("ks", "bn", "bm"))'
    elif cfg.swizzle:
        mesh = 'Mesh(("cta",), layout=(GN, GM), names=("bn", "bm"))'
    else:
        mesh = 'Mesh(("cta",), layout=(GM, GN), names=("bm", "bn"))'

    out_init = 'tf.zeros(Tensor[(SPLIT, M, N), "f32"])' if SK > 1 else 'tf.zeros(Tensor[(M, N), "bf16"])'
    src += f'''

@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", {topo}), Topology("thread", 384)))
class {cls}:
    @func
    def gemm({params}) -> {ret}:
        with {mesh} as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n={NI}, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = {out_init}
                m = cta.bm * BM
                n = cta.bn * BN
'''
    if two:
        src += f'''                with {COMPUTE} as _compute:
                    acc_g = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
                    acc_u = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
'''
        src += mainloop([("w_gate", "gate_ring", "acc_g"), ("w_up", "up_ring", "acc_u")], 0, "K")
        src += f'''                with {COMPUTE} as _compute:
                    g = tf.cast(tf.cast(acc_g, dtype="bf16"), dtype="f32")
                    u = tf.cast(tf.cast(acc_u, dtype="bf16"), dtype="f32")
                    act = tf.cast(g / (tf.exp(tf.neg(g)) + 1.0), dtype="bf16")
                    tile_out = tf.cast(tf.cast(act, dtype="f32") * u, dtype="bf16")
'''
    else:
        src += f'''                with {COMPUTE} as _compute:
                    acc = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
'''
        if SK > 1:
            src += mainloop([("w", "rhs_ring", "acc")], 0, "KS", k_base="cta.ks * KS")
        else:
            src += mainloop([("w", "rhs_ring", "acc")], 0, "K")
        src += f"                with {COMPUTE} as _compute:\n"
        if SK > 1:
            src += "                    tile_out = acc\n"
        elif epi == "residual":
            src += f'''                    mixed = tf.cast(acc, dtype="bf16")
                    r_dst = tf.empty(Tensor[(BM, BN), "bf16", {acc_layout(BN)}, "rmem"])
                    r_tile = tf.schedule((residual[m:m + BM, n:n + BN], r_dst), op=T.copy())
                    tile_out = tf.cast(tf.cast(r_tile, dtype="f32") + tf.cast(mixed, dtype="f32"), dtype="bf16")
'''
        else:
            src += '                    tile_out = tf.cast(acc, dtype="bf16")\n'
    dt = "f32" if SK > 1 else "bf16"
    width = BN
    if SK > 1:
        src += f'''                    tile_dst = tf.empty(Tensor[(1, BM, BN), "f32", (1, BM, BN)])
                    tile = tf.schedule((tf.reshape(tile_out, new_shape=(1, BM, BN)), tile_dst), op=T.copy())
                    out = tf.insert_slice(out, tile, (cta.ks, m, n))
                return out
'''
    else:
        src += f'''                    staged_dst = tf.empty(Tensor[(BM, {width}), "{dt}", (BM, {width}), "smem"])
                    staged = tf.schedule((tile_out, staged_dst), op=T.copy())
                with threads[0, :32] as _storer:
                    tile_dst = tf.empty(Tensor[(BM, {width}), "{dt}", (BM, {width})])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (m, n))
                return out
'''
    lhs = ("Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)" if BK == 64 else
           "Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3)")
    if linear == "gdn_ba":
        src = (src.replace(f'Plan row: runtime/plan.py, length {T}: {cfg}',
                           f'Config: runtime/kernels/gemm.py ba_splitk_kernel (split ba_split({T}) = {SK}): {cfg}')
               .replace("(f32 partial sums, then a reduce + residual pass), tiles launched M-fastest",
                        "(f32 partial sums, then reduce_ba), tiles launched N-fastest")
               .replace('Runtime kernel: runtime/kernels/gemm.py gemm_kernel(..., "splitk")',
                        f'Runtime kernel: runtime/kernels/gemm.py ba_splitk_kernel({T}, 5120, {SK}, BN={BN})'))
    return (src.replace("{RING}", ring(BN, BK)).replace("{RING_T}", ring_t(BN, BK)).replace("{LHS}", lhs)
            .replace("{KREP}", str(BK // 16)))


WS_HEADER = """\"\"\"Scheduled HIR: {linear} at prompt length {T} (hand warp-specialized kernel).

Implements: {what}
Runtime kernel: runtime/kernels/gemm_ws.py gemm_ws_kernel({T}, {n_call}, {K}, cfg, "{epi_ws}"),
  cfg = {cfg} (tag {tag}; runtime/plan.py ws_choice({T}, "{linear}"))
Finalized TIR: tir/gemm/{linear}_T{T}.py

Structure (as the kernel): {grid_txt}
384 threads: warpgroup 0 = producer, warp 0 issues every TMA load (the A tile 128 x 64 and
{w_txt} into {S}-deep 128B-swizzled rings){io_txt}.
Warpgroups 1-2 = consumers: WGMMA m64n{BN}k16 SS, W read K-major through a bitcast view,
warpgroup g owns rows [64g, 64g+64), {acc_txt}, k16 steps ascending; then
{epi_txt}.

Differences to the kernel:
- Synchronisation is not stated: the full/empty mbarrier ring (full count 32, empty count 8),
  the wgmma wait per k step, the named barriers of the consumer warpgroups and setmaxnreg
  40 / 232; TileFoundry derives the order from the data flow.
{diffs}\"\"\"
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = {M}
N = {N}
K = {K}
BM = {BM}
BN = {BN}
BK = {BK}
STAGES = {S}
GM = M // BM
GN = N // BN
UNITS = GM * GN
"""


def _indent(text, n):
    return "".join((" " * n + line if line.strip() else line) + "\n" for line in text.rstrip("\n").split("\n"))


def ws_source(linear, T, cfg, epi_ws):
    """Scheduled HIR of runtime/kernels/gemm_ws.py gemm_ws_kernel for one (linear, T)."""
    N, K, epi, what = LINEARS[linear]
    BM, BN, BK, S = cfg.BM, cfg.BN, cfg.BK, cfg.stages
    assert (BM, BK, cfg.split_k, cfg.CL) == (128, 64, 1, 1) and T % BM == 0, (linear, T, cfg)
    assert not (cfg.csk or cfg.direct or cfg.wbox or cfg.pdl or cfg.wait1 or cfg.debug), cfg
    n_call = N
    diffs = []
    if N % BN:
        N = -(-N // BN) * BN
        diffs.append(f"- N is {n_call}; the program is stated over {N} = {N // BN} whole tiles: weight rows >= {n_call}"
                     f" are TMA out-of-bounds zero fill and output columns >= {n_call} are clipped by the TMA store"
                     " (as in the kernel), so the valid region is the authored computation.")
    two, res, io = epi == "swiglu", epi == "residual", cfg.io
    GM, GN = T // BM, N // BN
    UNITS = GM * GN
    persist = cfg.grid > 0
    GRID = min(cfg.grid, UNITS) if persist else UNITS
    if persist:
        grid_txt = (f"persistent, {GRID} CTAs (one per SM); CTA p walks the tile list\n"
                    f"u = p, p + {GRID}, p + {2 * GRID}, ... < {UNITS}, M fastest (tile u = rows (u % {GM}) * 128,"
                    f" columns (u // {GM}) * {BN}),\nstated as the tile loop `for u in range(cta.p, UNITS, NSM)`.")
        diffs.append("- Persistent loop: the kernel's ring slot sequence continues across a CTA's tiles (slot =\n"
                     "  (it * KT + k) % stages), the producer runs ahead into the next tile while the consumers\n"
                     "  run the epilogue, and a tile's TMA store is waited for only before the staging tile is\n"
                     "  written again. Here the ring is declared in the tile loop (slot = k % stages restarts at\n"
                     "  each tile); the overlap across tiles is left to lowering.")
    else:
        grid_txt = (f"one CTA per (128 x {BN}) output tile, {UNITS} CTAs, tile order M fastest\n"
                    f"(CTA b = rows (b % {GM}) * 128, columns (b // {GM}) * {BN}).")
    w_txt = f"the W_gate and W_up tiles {BN} x 64" if two else f"the W tile {BN} x 64"
    io_txt = (";\nwarp 1 (IO warp) TMA-loads the residual tile R and issues the output TMA store" if io else "")
    acc_txt = "two f32 accumulators (gate, up)" if two else "one f32 accumulator"
    if two:
        epi_txt = ("the SwiGLU epilogue g = bf16(acc_g), u = bf16(acc_u), bf16(silu(g)), bf16(. * u) in the\n"
                   "accumulator registers, the tile staged in shared memory and TMA-stored by a consumer thread")
    elif res and io:
        epi_txt = ("the residual epilogue bf16(R + bf16(acc)) with R read from the IO warp's shared tile, the tile\n"
                   "staged in shared memory and TMA-stored by the IO warp")
    elif res:
        epi_txt = ("the residual epilogue bf16(R + bf16(acc)) with R loaded from global into the accumulator\n"
                   "layout, the tile staged in shared memory and TMA-stored by a consumer thread")
    else:
        epi_txt = "bf16(acc) staged in shared memory and TMA-stored by a consumer thread"
    if cfg.pk:
        diffs.append("- pk: the kernel's epilogue casts are packed bf16x2 conversions (T.copy fragment ->\n"
                     "  fragment)" + (" and the SwiGLU reciprocal is a non-volatile rcp.approx.ftz" if two else "")
                     + "; same roundings. Stated as plain casts" + (" and g / (exp(-g) + 1)." if two else "."))
    elif two:
        diffs.append("- The SwiGLU reciprocal (tl fast_rcp, rcp.approx) is stated as g / (exp(-g) + 1).")
    if cfg.amerge:
        diffs.append("- amerge: the kernel compiles with tl.enable_aggressive_shared_memory_merge, so the output\n"
                     "  staging tile aliases the A ring (one tile per CTA); here the staging tile is its own buffer.")
    if io:
        diffs.append("- io: the kernel's IO warp TMA-loads R into the output staging tile itself, the consumers\n"
                     "  read R from it and overwrite it with the result, and the IO warp stores that tile; here R\n"
                     "  lands in its own shared tile and the result in a second one (both TMA ops on warp 1).")
    stg_swz = (BN * 2) % 128 == 0 and BN % 64 == 0
    if not stg_swz:
        diffs.append(f"- Staging tile: plain (128, {BN}) bf16 (the kernel swizzles it only when BN is a multiple of 64).")
    stage_layout = (f"Layout(((16, 8), ({BN // 64}, 64)), ((512, 64), ({BM * 64}, 1))) | Swizzle(3, 4, 3)"
                    if stg_swz else f"(BM, BN)")

    import dataclasses
    dflt = type(cfg)()
    cfg_txt = "WsConfig(" + ", ".join(f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg)
                                      if getattr(cfg, f.name) != getattr(dflt, f.name) or f.name in ("BN", "stages")) + ")"
    src = WS_HEADER.format(linear=linear, T=T, what=what, n_call=n_call, K=K, cfg=cfg_txt, epi_ws=epi_ws, tag=cfg.tag(),
                           grid_txt=grid_txt, w_txt=w_txt, S=S, io_txt=io_txt, BN=BN, acc_txt=acc_txt, epi_txt=epi_txt,
                           diffs="\n".join(diffs) + "\n", M=T, N=N, BM=BM, BK=BK)
    if persist:
        src += f"NSM = {GRID}\n"
    cls = f"SCHED_{linear.upper()}_T{T}"
    params = 'a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"]'
    if two:
        params = 'a: Tensor[(M, K), "bf16"], w_gate: Tensor[(N, K), "bf16"], w_up: Tensor[(N, K), "bf16"]'
    if res:
        params += ', residual: Tensor[(M, N), "bf16"]'
    mesh = ('Mesh(("cta",), layout=(NSM,), names=("p",))' if persist
            else 'Mesh(("cta",), layout=(GN, GM), names=("bn", "bm"))')
    src += f'''

@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", {"NSM" if persist else "UNITS"}), Topology("thread", 384)))
class {cls}:
    @func
    def gemm({params}) -> Tensor[(M, N), "bf16"]:
        with {mesh} as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n={BN}, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
'''
    tile = ""
    if io:
        tile += f'''                with threads[0, 32:64] as _io:
                    r_s = tf.schedule((residual[m:m + BM, n:n + BN], tf.empty(Tensor[(BM, BN), "bf16", (BM, BN), "smem"])), op=T.copy_async_tensor())
'''
    if two:
        tile += f'''                with {COMPUTE} as _compute:
                    acc_g = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
                    acc_u = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
'''
        tile += mainloop([("w_gate", "gate_ring", "acc_g"), ("w_up", "up_ring", "acc_u")], 0, "K")
        tile += f'''                with {COMPUTE} as _compute:
                    g = tf.cast(tf.cast(acc_g, dtype="bf16"), dtype="f32")
                    u = tf.cast(tf.cast(acc_u, dtype="bf16"), dtype="f32")
                    act = tf.cast(g / (tf.exp(tf.neg(g)) + 1.0), dtype="bf16")
                    tile_out = tf.cast(tf.cast(act, dtype="f32") * u, dtype="bf16")
'''
    else:
        tile += f'''                with {COMPUTE} as _compute:
                    acc = tf.zeros(Tensor[(BM, BN), "f32", {acc_layout(BN)}, "rmem"])
'''
        tile += mainloop([("w", "rhs_ring", "acc")], 0, "K")
        tile += f"                with {COMPUTE} as _compute:\n"
        if res:
            r_src = "r_s" if io else "residual[m:m + BM, n:n + BN]"
            tile += f'''                    mixed = tf.cast(acc, dtype="bf16")
                    r_tile = tf.schedule(({r_src}, tf.empty(Tensor[(BM, BN), "bf16", {acc_layout(BN)}, "rmem"])), op=T.copy())
                    tile_out = tf.cast(tf.cast(r_tile, dtype="f32") + tf.cast(mixed, dtype="f32"), dtype="bf16")
'''
        else:
            tile += '                    tile_out = tf.cast(acc, dtype="bf16")\n'
    storer = "threads[0, 32:64] as _io" if io else "threads[1, :32] as _storer"
    tile += f'''                    staged = tf.schedule((tile_out, tf.empty(Tensor[(BM, BN), "bf16", {stage_layout}, "smem"])), op=T.copy())
                with {storer}:
                    tile = tf.schedule((staged, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (m, n))
'''
    if persist:
        src += "                for u in range(cta.p, UNITS, NSM):\n"
        src += "                    m = (u % GM) * BM\n                    n = (u // GM) * BN\n"
        src += _indent(tile, 4)
    else:
        src += "                m = cta.bm * BM\n                n = cta.bn * BN\n" + tile
    src += "                return out\n"
    return (src.replace("{RING}", ring(BN, BK)).replace("{RING_T}", ring_t(BN, BK))
            .replace("{LHS}", "Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)")
            .replace("{KREP}", str(BK // 16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    args = ap.parse_args()
    sdir, tdir = ROOT / "scheduled/gemm", ROOT / "tir/gemm"
    sdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    fails = 0
    for T in args.lengths:
        p = plan.for_length(T)
        for linear in LINEARS:
            if args.only and linear != args.only:
                continue
            if linear == "gdn_ba":
                from runtime.kernels.gemm import GemmConfig, ba_bn, ba_split
                cfg = GemmConfig(BM=128, BN=ba_bn(T), BK=64, stages=2, swizzle=0, split_k=ba_split(T))
            else:
                cfg = p.gemm[linear]
            path = sdir / f"{linear}_T{T}.py"
            ws = plan.ws_choice(T, linear) if linear != "gdn_ba" else None
            if ws:
                cfg = ws[0]
                path.write_text(ws_source(linear, T, *ws))
            else:
                path.write_text(gemm_source(linear, T, cfg))
            out = tdir / f"{linear}_T{T}.py"
            r = subprocess.run([TILEFOUNDRY, "schedule", "finalize", str(path), str(out)],
                               capture_output=True, text=True)
            ok = r.returncode == 0
            fails += not ok
            print(f"T={T} {linear:8s} {cfg.tag():28s} finalize {'OK' if ok else 'FAIL'}"
                  + ("" if ok else "\n    " + r.stderr.strip().replace("\n", "\n    ")[:900]), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
