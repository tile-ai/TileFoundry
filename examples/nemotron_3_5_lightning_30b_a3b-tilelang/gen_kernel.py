#!/usr/bin/env python
"""Emit mega_kernel.py: the whole decode step as one persistent cooperative kernel.

TileLang's builder rewrites every `for` in a `@T.prim_func` into a device loop --
`range` is overridden to `T.serial` -- and it does not bind a nested `def`. So a
kernel that wants trace-time repetition (a prefetch prologue, a fixed number of
accumulator registers, one body per stage kind) has to have that repetition
written out. This file is where the repetition lives; `mega_kernel.py` is what
TileLang reads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import model

H, V = model.H, model.V
MH, MD, SS, NG, MI, CONV, KER, WIN, PROJ = (model.MH, model.MD, model.SS, model.NG,
                                            model.MI, model.CONV, model.KER, model.WIN,
                                            model.PROJ)
GRP, HPG = model.GRP, model.HPG
HQ, HKV, DH, QP, KVP, GQA = (model.HQ, model.HKV, model.DH, model.QP,
                             model.KVP, model.GQA)
E, KTOP, I, IS = model.E, model.K, model.I, model.IS
KINDS = model.LAYER_KINDS
NL = len(KINDS)

#: The grid is two-dimensional because the mesh is: `bx` is a query head and
#: `by` a worker over the context, which is what the short attention split
#: needs, and a one-dimensional grid can only produce that pair with a `//` and
#: a `%` -- and a bulk copy whose address carries either of those falls off the
#: one-dimensional path onto a descriptor (`kbench/tma2d.py` measures which
#: address shapes survive). Everything that does not care sees the flat index.
#: 33 x 4 = 132, the SM count, so no SM is left out; `bx` is still the query
#: head it was, and the 33rd column simply has none: its trip count is zero.
NHEAD = 33
NW = 4
CTAS = NHEAD * NW
#: The flat index spelled out, for the places that must keep the block index
#: visible inside a bulk copy's address rather than behind a `let`.
CTAX = f"(bx + by * {NHEAD})"
THREADS = 256
RB = THREADS // 32
#: Two stages of the widest tile (kd = 4096) and nothing over: what the arena
#: gives back is what attention's operand buffer costs, and every gemv keeps the
#: stage count it had. Measured free at 64512 back when it was 90000.
ARENA = 2 * (THREADS // 32) * 4096
#: f32 shared: 32 rows of Q or of the block's scores, then the small MoE slots.
FROW = 33
FARENA = FROW * 128
ABLK = 128
KVROW = HKV * DH
#: How many ABLK blocks the K/V views declare. The cache holds 262144 positions
#: at most, which is 2048 of them; the tail is two.
CBLK = 4096
#: Where the two attention splits cross: per-CTA work is nblk blocks x 1 head
#: one way and ceil(nblk/132) blocks x 32 heads the other, equal at nblk = 32.
#: Where the two attention splits change over. The same number the authored
#: HIR dispatches on (`attention.py`'s CROSSOVER), read off the per-unit work
#: of the two placements rather than tuned: at ctx_full = 2048 walking the
#: whole context with one query head passes carrying all 32 heads over a
#: stripe of it. Measured on the card the crossing is nearer 3000 -- at 2048
#: the two are within 2% of each other -- so the analytic boundary is used.
HSPLIT = int(__import__('os').environ.get('NEMO_HSPLIT', 2048))
UPT = THREADS // 8
RING = 4
#: The arena's tail holds the six `mid` vectors while the MoE's down half runs.
MIDOFF_VAL = ARENA - KTOP * I

_KIND_ID = {"linear_attention": 0, "full_attention": 1, "moe": 2}
_LAYER_META, _seen = [], {0: 0, 1: 0, 2: 0}
for _k in KINDS:
    _LAYER_META.append((_KIND_ID[_k], _seen[_KIND_ID[_k]]))
    _seen[_KIND_ID[_k]] += 1
N_MAMBA, N_ATTN, N_MOE = _seen[0], _seen[1], _seen[2]

NUM = {
    'win': N_MAMBA * PROJ * H, 'wout': N_MAMBA * H * MI,
    'wqkv': N_ATTN * (QP + 2 * KVP) * H, 'wo': N_ATTN * H * QP,
    'wrt': N_MOE * E * H, 'wup': N_MOE * E * I * H,
    'wdn': N_MOE * E * H * I, 'wsu': N_MOE * IS * H,
    'wsd': N_MOE * H * IS, 'whead': V * H,
}


def stages_for(kd: int, arena=None) -> int:
    return max(2, min(6, (ARENA if arena is None else arena) // (RB * kd)))


#: Stage labels a profiling build records, in the order they are probed.
PROBES: list[str] = []
PROF = False


def probe(e, label: str):
    """Charge the cycles since the last probe to `label`, on CTA 0.

    Only built when `gen_kernel.py --prof` asks for it: the kernel that ships
    has no probes in it. The barrier is what makes the reading mean "the whole
    CTA got here", and the counter is accumulated because the stage runs once
    per layer of its kind.
    """
    if not PROF:
        return
    if label not in PROBES:
        PROBES.append(label)
    k = PROBES.index(label)
    e.block("if cta == 0:")
    e("T.sync_threads()")
    e.block("if tid == 0:")
    e(f"prof[{k}] = prof[{k}] + T.call_extern(\"int64\", \"clock64\") - prof[NPROBE]",
      f"prof[NPROBE] = T.call_extern(\"int64\", \"clock64\")").end().end()


class Emit:
    """A little indented-source writer."""

    def __init__(self, indent=0):
        self.lines = []
        self.ind = indent

    def __call__(self, *lines):
        for line in lines:
            self.lines.append(("    " * self.ind + line) if line else "")
        return self

    def block(self, header):
        self(header)
        self.ind += 1
        return self

    def end(self, n=1):
        self.ind -= n
        return self

    def text(self):
        return "\n".join(self.lines)


def gemv(e: Emit, w, wnum, unit, usize, n, kd, xoff, mode, doff="0",
         nunit=1, arena=None, xsrc="fs", scale=None):
    """Stream this CTA's rows of a `(n, kd)` matrix and dot each with x.

    The matrix is unit number `unit` of `w`, and one unit is `usize` elements --
    a layer for most weights, a (layer, expert) pair for the routed ones. The two
    expert banks hold 1.47e10 elements each, past what an int32 offset can name,
    so their address is built in int64 out of two clamped int32 pieces rather
    than computed in int32 and then widened.

    With `nunit > 1` one pass walks that many units back to back -- the six
    routed experts of a MoE layer are one loop, not six. `unit`, `doff` and
    `xoff` may then be callables of the unit index, which is an expression in the
    tile counter. It matters twice over: a two-tile pass pays a cold memory
    pipeline for both of its tiles, and six copies of an unrolled body are six
    times the instruction bytes to fetch.

    `xsrc` is where the vector lives -- shared f32 by default, or `sm` when it is
    the bf16 tail of the arena, which is how the second half of a MoE layer reads
    the six `mid` vectors without a shared allocation of its own.

    The tile that would run past the last row is slid back rather than masked, so
    every bulk copy is one static size; the rows it repeats are recomputed to the
    same bits by the CTA that owns them anyway. The clamp is also what makes the
    bound *provable*: a copy whose bounds cannot be proved falls off the
    one-dimensional bulk path onto a descriptor, which splits one 43 KB tile into
    84 loads of 512 B.

    Nothing here binds a local name: TileLang's builder scopes a binding to the
    frame it was made in, and these addresses are read again inside the `if` that
    issues the next tile.
    """
    tile = RB * kd
    nr = (n + CTAS - 1) // CTAS
    ntu = (nr + RB - 1) // RB
    ntile = nunit * ntu
    ns = min(stages_for(kd, arena), ntile)
    big = wnum > 2 ** 31 - 1
    maxunit = (wnum // usize - 1) if usize else 0
    fixed = {}

    def at_tile(v, t):
        """`v` for tile `t`: a plain string, or a callable of the unit index."""
        if callable(v):
            return v(f"(({t}) // {ntu})" if nunit > 1 else "0")
        return v

    def row(t):
        b = f"({t}) % {ntu}" if nunit > 1 else f"({t})"
        # `CTAX`, not `cta`: this expression lands inside a bulk copy address.
        return f"T.max(T.min({CTAX} * {nr} + ({b}) * {RB}, {n - RB}), 0)"

    def off(t):
        u = at_tile(unit, t)
        if big:
            return (f"(T.Cast(\"int64\", T.max(T.min({u}, {maxunit}), 0)) * {usize}"
                    f" + T.Cast(\"int64\", {row(t)}) * {kd})")
        base = f"({u}) * {usize} + " if usize else ""
        return f"T.max(T.min({base}{row(t)} * {kd}, {wnum - tile}), 0)"

    def xread(t, off_in_row):
        x = at_tile(xoff, t)
        idx = f"{x} + {off_in_row}"
        return (f"T.Cast(\"float32\", sm[{idx}])" if xsrc == "sm" else f"fs[{idx}]")

    e(f"# --- gemv {n} x {kd}: {ntile} tiles of {RB} rows, {ns} in flight"
      + (f", {nunit} units in one pass" if nunit > 1 else ""))
    # One CTA-wide barrier between prologue issues. Without it TileLang's sync
    # pass sees a shared-memory hazard between two issues, and the sync it adds
    # to fix it lands *inside* the single-lane elect that `tma_copy` lowers to --
    # where exactly one thread would reach it.
    for j in range(ns):
        e("T.sync_threads()")
        e(f"T.tma_copy({w}[{off(j)}:{off(j)} + {tile}],"
          f" sm[{j * tile}:{(j + 1) * tile}], barrier=bar[{j}])")
        e.block("if T.shuffle_elect(THREADS):")(f"T.barrier_arrive(bar[{j}])").end()
    e("T.sync_threads()")
    e.block(f"for _t in T.serial({ntile}):")
    e(f"T.mbarrier_wait_parity(bar[_t % {ns}], (_t // {ns}) % 2)", "acc[0] = 0.0")
    e.block(f"for _i in T.serial({kd // 64}):")
    e(f"acc[0] += {xread('_t', f'lane * 2 + _i * 64')} * T.Cast(\"float32\","
      f" sm[(_t % {ns}) * {tile} + warp * {kd} + lane * 2 + _i * 64])",
      f"acc[0] += {xread('_t', f'lane * 2 + _i * 64 + 1')} * T.Cast(\"float32\","
      f" sm[(_t % {ns}) * {tile} + warp * {kd} + lane * 2 + _i * 64 + 1])").end()
    # A warp holds one whole row, so its own lanes are the whole reduction. The
    # eight sums are handed to warp 0 to store, because eight lanes of one warp
    # writing eight neighbouring rows is one store transaction, where eight
    # warps writing one row each is eight of them into the same sector.
    e("red[tid] = acc[0]", "T.sync_threads()")
    e.block(f"if tid < {RB}:")("acc[0] = 0.0")
    e.block("for _q in T.serial(32):")("acc[0] += red[tid * 32 + _q]").end().end()
    # Store only the rows this tile owns: at or past where it would have started
    # had it not been slid back, and inside this CTA's run. The upper guard keeps
    # a tile off the next CTA's rows -- it computes them to the same bits, but
    # writing them would race with that CTA *reading* them, which the MoE's down
    # half does with no barrier in between because it needs none. The lower guard
    # keeps the slid-back tile off the rows its predecessor already did, which an
    # idempotent store would not care about but an accumulating one would.
    r = f"({row('_t')} + tid)"
    bt = f"(_t) % {ntu}" if nunit > 1 else "(_t)"
    e.block(f"if tid < {RB}:")
    e.block(f"if {r} >= cta * {nr} + ({bt}) * {RB}:")
    e.block(f"if {r} < T.min((cta + 1) * {nr}, {n}):")
    d = at_tile(doff, "_t")
    if mode == "bf":
        e(f"scratch[{d} + {r}] = bf(acc[0])")
    elif mode in ("f32", "raw"):
        e(f"scratch[{d} + {r}] = acc[0]")
    elif mode == "relu2":
        e("acc[0] = T.max(bf(acc[0]), 0.0)",
          f"scratch[{d} + {r}] = bf(acc[0] * acc[0])")
    elif mode == "logits":
        e(f"logits[{r}] = bf(acc[0])")
    elif mode == "local":
        # Into the CTA's own shared slice, not global: six experts accumulating
        # into a global row would be six dependent read-modify-writes per tile,
        # each of them a full memory round trip on the critical path.
        e(f"fs[{d} + {r} - cta * {nr}] = fs[{d} + {r} - cta * {nr}]"
          f" + {at_tile(scale, '_t')} * bf(acc[0])")
    elif mode == "localraw":
        e(f"fs[{d} + {r} - cta * {nr}] = acc[0]")
    elif mode == "qkv":
        e(f"scratch[{d} + {r}] = bf(acc[0])")
        e.block(f"if {r} >= {QP}:")
        e.block(f"if {r} < {QP + KVP}:")
        e(f"k_tail[cur_pos * {KVROW} + ({r} - {QP})] = T.Cast(\"bfloat16\", acc[0])").end()
        e.block("else:")
        e(f"v_tail[cur_pos * {KVROW} + ({r} - {QP + KVP})] ="
          f" T.Cast(\"bfloat16\", acc[0])").end().end()
    else:
        raise ValueError(mode)
    e.end().end().end()
    e("T.sync_threads()")
    e.block(f"if _t + {ns} < {ntile}:")
    e(f"T.tma_copy({w}[{off(f'_t + {ns}')}:{off(f'_t + {ns}')} + {tile}],"
      f" sm[(_t % {ns}) * {tile}:(_t % {ns}) * {tile} + {tile}], barrier=bar[_t % {ns}])")
    e.block("if T.shuffle_elect(THREADS):")(f"T.barrier_arrive(bar[_t % {ns}])")
    e.end().end()
    e.end()
    for s_ in range(ns):
        if ((ntile - s_ + ns - 1) // ns) % 2 == 1:
            e.block("if T.shuffle_elect(THREADS):")(f"T.barrier_arrive(bar[{s_}])").end()
            e(f"T.mbarrier_wait_parity(bar[{s_}], 1)")


def bulk(e: Emit, pairs):
    """Bulk-copy each `(src slice, element count, arena offset)` into the arena.

    The B/C half of a Mamba layer's convolution wants five two-byte values per
    channel, four elements apart, and every CTA wants all 2048 channels: read
    straight from global that is eight sectors per load instruction. Read once in
    bulk and the strided part happens in shared memory instead.
    """
    for k, (src, num, off) in enumerate(pairs):
        e("T.sync_threads()")
        e(f"T.tma_copy({src}, sm[{off}:{off} + {num}], barrier=bar[{k}])")
        e.block("if T.shuffle_elect(THREADS):")(f"T.barrier_arrive(bar[{k}])").end()
    e("T.sync_threads()")
    for k in range(len(pairs)):
        e(f"T.mbarrier_wait_parity(bar[{k}], 0)")
    # Leave every barrier on the phase the next gemv's prologue expects.
    for k in range(len(pairs)):
        e.block("if T.shuffle_elect(THREADS):")(f"T.barrier_arrive(bar[{k}])").end()
        e(f"T.mbarrier_wait_parity(bar[{k}], 1)")


def spread(e: Emit, n, body_lines, var="_j"):
    """Run the body once per index in [0, n), spread over the CTA's threads."""
    reps = (n + THREADS - 1) // THREADS
    # `T.unroll`, not `T.serial`: with one CTA per SM there are eight warps to
    # hide a global load behind, so what hides it is the next load, and the
    # loads only issue back to back if the loop body is unrolled.
    e.block(f"for _i in T.unroll({reps}):")
    e(f"{var} = _i * {THREADS} + tid")
    if n % THREADS:
        e.block(f"if {var} < {n}:")
        e(*body_lines)
        e.end()
    else:
        e(*body_lines)
    e.end()


def cta_sum(e: Emit, value, out):
    """CTA-wide sum of `value`, left in `out` (a shared scalar slot).

    A warp folds its own 32 lanes with shuffles, so only one partial per warp
    reaches shared memory and the serial tail is eight adds rather than 256.
    """
    e(f"acc[0] = {value}", "acc[0] = T.warp_reduce_sum(acc[0])")
    e.block("if lane == 0:")("red[warp] = acc[0]").end()
    e("T.sync_threads()")
    e.block("if tid == 0:")
    e("acc[0] = " + " + ".join(f"red[{w}]" for w in range(THREADS // 32)))
    e(f"{out} = acc[0]").end()
    e("T.sync_threads()")


def rmsnorm(e: Emit, gamma_expr):
    """The published RMSNorm of the hidden row into fs[0:H], redundantly per CTA.

    The sum of squares is not computed here: whoever last wrote the row -- the
    embedding, or the residual add that closed the layer before -- had every
    element in a register already and left the sum in `sml[SUMSLOT]`, so this is
    one pass over shared memory and no reduction at all.
    """
    e(f"fv[0] = T.rsqrt(sml[SUMSLOT] / {float(H)} + EPS)")
    spread(e, H, [f"fs[_j] = bf(bf(T.Cast(\"float32\", hs[_j]) * fv[0])"
                  f" * {gamma_expr})"])
    e("T.sync_threads()")


def residual(e: Emit):
    """h += mix, every CTA holding its own copy of the whole row.

    The hidden row lives in shared memory, one copy per CTA, for the whole step:
    every CTA needs all of it every layer, and dividing it would cost a barrier
    to put it back together. So the only thing this reads from global memory is
    the mix the layer just produced. The sum of squares the next pre-norm wants
    is taken here, where the new value is already in a register.
    """
    e("acc[0] = 0.0")
    spread(e, H, ["fv[1] = T.Cast(\"float32\", hs[_j]) + scratch[S_MIX + _j]",
                  "hs[_j] = bf(fv[1])",
                  "acc[0] += T.Cast(\"float32\", hs[_j])"
                  " * T.Cast(\"float32\", hs[_j])"])
    cta_sum(e, "acc[0]", "sml[SUMSLOT]")


def copy_to_fs(e: Emit, src, n):
    spread(e, n, [f"fs[_j] = scratch[{src} + _j]"])
    e("T.sync_threads()")


# ---------------------------------------------------------------------------
# The three layer kinds.
# ---------------------------------------------------------------------------

def mamba(e: Emit):
    e("# ============================== Mamba-2")
    probe(e, "prenorm")
    gemv(e, "win", NUM["win"], "at", 27697152, PROJ, H, 0, "bf", "S_PROJ")
    probe(e, "m.in_proj")
    e("T.sync_grid()")
    probe(e, "m.bar")
    e("conv_in = T.make_tensor_from_addr(ptrs[P_CONV_IN + at], (CONV * WIN,), \"bfloat16\")",
      "conv_out = T.make_tensor_from_addr(ptrs[P_CONV_OUT + at], (CONV * WIN,), \"bfloat16\")",
      "ssm_in = T.make_tensor_from_addr(ptrs[P_SSM_IN + at], (MI * SS,), \"float32\")",
      "ssm_out = T.make_tensor_from_addr(ptrs[P_SSM_OUT + at], (MI * SS,), \"float32\")")
    e("# B and C are 2048 channels every CTA needs some of, so every CTA",
      "# convolves all of them rather than pay a barrier to share them. Their",
      "# window, weights and bias come in as three bulk copies first.")
    bulk(e, [(f"conv_in[{MI * WIN}:{CONV * WIN}]", (CONV - MI) * WIN, 0),
             (f"convw[(at * CONV + MI) * KER:(at * CONV + CONV) * KER]",
              (CONV - MI) * KER, (CONV - MI) * WIN),
             (f"convb[at * CONV + MI:at * CONV + CONV]", CONV - MI,
              (CONV - MI) * (WIN + KER))])
    conv_body = ["acc[0] = 0.0"]
    for j in range(KER - 1):
        conv_body.append(
            f"acc[0] += T.Cast(\"float32\", sm[_j * WIN + {j}])"
            f" * T.Cast(\"float32\", sm[{(CONV - MI) * WIN} + _j * KER + {j}])")
    conv_body += [
        "acc[0] += scratch[S_PROJ + MI + MI + _j]"
        f" * T.Cast(\"float32\", sm[{(CONV - MI) * WIN} + _j * KER + {KER - 1}])",
        "acc[0] = bf(acc[0] + T.Cast(\"float32\","
        f" sm[{(CONV - MI) * (WIN + KER)} + _j]))",
        "fs[F_BC + _j] = bf(acc[0] * T.sigmoid(acc[0]))",
    ]
    spread(e, 2 * NG * SS, conv_body)
    probe(e, "m.conv_bc")

    e("# the new convolution window: one slice of channels per CTA")
    nc = (CONV + CTAS - 1) // CTAS
    e.block(f"for _i in T.serial({(nc + THREADS - 1) // THREADS}):")
    e(f"_c = cta * {nc} + _i * {THREADS} + tid")
    e.block("if _c < CONV:")
    for j in range(WIN - 1):
        e(f"conv_out[_c * WIN + {j}] = conv_in[_c * WIN + {j + 1}]")
    e(f"conv_out[_c * WIN + {WIN - 1}] = T.Cast(\"bfloat16\", scratch[S_PROJ + MI + _c])")
    e.end().end()
    e("T.sync_threads()")
    probe(e, "m.window")

    e("# the SSM step: eight threads per (head, channel) unit")
    nu = (MI + CTAS - 1) // CTAS
    e.block(f"for _ub in T.serial({(nu + UPT - 1) // UPT}):")
    e("_ul = tid // 8", "_part = tid % 8", f"_u = cta * {nu} + _ub * {UPT} + _ul")
    e.block("if _part == 0:")
    e.block("if _u < MI:")
    e("acc[0] = 0.0")
    for j in range(KER - 1):
        e(f"acc[0] += T.Cast(\"float32\", conv_in[_u * WIN + {j}])"
          f" * T.Cast(\"float32\", convw[(at * CONV + _u) * KER + {j}])")
    e("acc[0] += scratch[S_PROJ + MI + _u]"
      f" * T.Cast(\"float32\", convw[(at * CONV + _u) * KER + {KER - 1}])",
      "acc[0] = bf(acc[0] + T.Cast(\"float32\", convb[at * CONV + _u]))",
      "fs[F_XV + _ul] = bf(acc[0] * T.sigmoid(acc[0]))")
    e.end().end()
    e("T.sync_threads()", "red[tid] = 0.0")
    e.block("if _u < MI:")
    e("fv[0] = scratch[S_PROJ + MI + CONV + _u // MD]"
      " + T.Cast(\"float32\", mscal[(at * 3 + 1) * MH + _u // MD])",
      "fv[0] = T.if_then_else(fv[0] > 20.0, fv[0], T.log(1.0 + T.exp(fv[0])))",
      "fv[0] = T.max(bf(fv[0]), bf(DTMIN))",
      "fv[1] = T.exp(fv[0] * (0.0 - T.exp(T.Cast(\"float32\","
      " mscal[at * 3 * MH + _u // MD]))))",
      "fv[2] = fs[F_XV + _ul]",
      "acc[0] = 0.0")
    # The eight threads of a unit interleave the state rather than taking a
    # block each, so one instruction reads eight neighbouring floats instead of
    # eight 64-byte-apart ones. Which of the 128 a thread takes does not matter:
    # the whole row is summed either way.
    e.block(f"for _q in T.unroll({SS // 8}):")
    e(f"fv[3] = ssm_in[_u * SS + _q * 8 + _part] * fv[1]"
      f" + bf(bf(fv[0] * fs[F_BC + (_u // MD // HPG) * SS + _q * 8 + _part])"
      f" * fv[2])",
      f"ssm_out[_u * SS + _q * 8 + _part] = fv[3]",
      f"acc[0] += bf(fv[3])"
      f" * fs[F_BC + NG * SS + (_u // MD // HPG) * SS + _q * 8 + _part]").end()
    e("red[tid] = acc[0]").end()
    e("T.sync_threads()")
    e.block("if _part == 0:")
    e.block("if _u < MI:")
    e("acc[0] = 0.0")
    e.block("for _q in T.serial(8):")("acc[0] += red[tid + _q]").end()
    e("scratch[S_Y + _u] = bf(bf(acc[0]) + bf(fs[F_XV + _ul]"
      " * T.Cast(\"float32\", mscal[(at * 3 + 2) * MH + _u // MD])))")
    e.end().end()
    e("T.sync_threads()").end()
    probe(e, "m.scan")
    e("T.sync_grid()")
    probe(e, "m.bar")

    e("# the gated group norm, then out_proj. There are as many groups as warps,",
      "# so a warp owns a whole group and the norm never leaves it: no shared",
      "# scalar, no barrier, and the eight groups run at once instead of in turn.")
    assert NG == THREADS // 32 and GRP % 32 == 0
    e("acc[0] = 0.0")
    e.block(f"for _i in T.unroll({GRP // 32}):")
    e(f"_j = warp * {GRP} + _i * 32 + lane",
      "fs[_j] = scratch[S_Y + _j] * (scratch[S_PROJ + _j]"
      " * T.sigmoid(scratch[S_PROJ + _j]))",
      "acc[0] += fs[_j] * fs[_j]").end()
    e("acc[0] = T.warp_reduce_sum(acc[0])",
      f"fv[0] = T.rsqrt(acc[0] / {float(GRP)} + EPS)")
    e.block(f"for _i in T.unroll({GRP // 32}):")
    e(f"_j = warp * {GRP} + _i * 32 + lane",
      "fs[_j] = bf(T.Cast(\"float32\", ggdn[at * MI + _j])"
      " * bf(fs[_j] * fv[0]))").end()
    e("T.sync_threads()")
    probe(e, "m.gnorm")
    gemv(e, "wout", NUM["wout"], "at", 11010048, H, MI, 0, "bf", "S_MIX")
    probe(e, "m.out_proj")
    e("T.sync_grid()")
    probe(e, "m.bar")
    residual(e)
    probe(e, "residual")


def kv_load(e: Emit, which: str, dst: str, col: str, tag: str, idx: str = "_bi"):
    """One ABLK block of one KV group's K, or V, into an operand buffer.

    Which buffer a block reads is decided on the block index: a block starts on
    a multiple of ABLK and so does `ctx_full`, so a block is wholly in the
    whole-block view or wholly in the remainder. Rather than branch on it -- a
    branch around a copy is a branch around the barrier the copy needs, and
    TileLang's sync pass will lift that barrier out of it -- the choice is made
    on the address, which is a select on an integer.

    A group is a column slice, so the copy is 256 contiguous bytes a row on a
    512-byte stride. That costs 13% against the full-width block
    (`kbench/blkloop.py`, 3.705 against 4.181 TB/s) and buys K and V both
    resident in 64 KB, which is what the arena leaves. The alternative that
    keeps the full width in the same 64 KB reads K, computes, then lands V on
    top of it, and pays 1272 ns a block against this shape's 979.
    """
    up = which.upper()
    e(f"_s{tag} = T.make_tensor_from_addr(T.if_then_else({idx} < _cfb,"
      f" ptrs[P_{up}C + at], ptrs[P_{up}T + at]),"
      f" (CBLK, {ABLK}, {KVROW}), \"bfloat16\")",
      f"_i{tag} = T.max(T.min(T.if_then_else({idx} < _cfb, {idx}, {idx} - _cfb),"
      f" {CBLK - 1}), 0)")
    e(f"T.copy(_s{tag}[_i{tag}, :, {col}:{col} + {DH}], {dst},"
      " prefer_instruction=\"cp_async\")")


def attn_by_head(e: Emit):
    """A query head per `bx`, a quarter of the context per `by`.

    While the context is short there are not enough blocks of it to divide over
    the grid -- at one block, the context split leaves 127 CTAs at a barrier
    while the first one carries all 32 heads. The head axis has 32 units in it
    whatever the length is, so this takes the head as the outer axis and splits
    what context there is four ways under it.

    One query head is one row, and one row is no work for a tensor core: this
    arm stays on the CUDA cores. What it shares with the long arm is the pair of
    operand buffers, and a CTA here wants exactly one KV group of them -- its
    head's -- so the column slice the long arm loads for its own reasons is the
    slice this one wanted anyway.

    Threads split by position for the scores and by dimension for the
    accumulate, two to each, so a warp's lanes differ in whichever axis is
    contiguous in shared memory at the time.
    """
    e("# ---- a head per bx, a quarter of the context per by")
    e(f"_nb = T.if_then_else(bx < 32, T.ceildiv(T.max(T.ceildiv(ctx_all, {ABLK}) - by, 0), NW), 0)",
      f"_grp = (bx // {GQA}) * {DH}")
    e.block("if tid == 0:")
    e(f"sml[{4 * HQ + 1}] = NEG", f"sml[{4 * HQ + 2}] = 0.0").end()
    e("av[0] = 0.0")
    e("T.sync_threads()")
    e.block("for _b in T.serial(_nb):")
    e("_bi = by + _b * NW",
      f"iv[0] = T.max(T.min(ctx_all - _bi * {ABLK}, {ABLK}), 0)")
    kv_load(e, "k", "ks", "_grp", "hk")
    kv_load(e, "v", "vs", "_grp", "hv")
    e("T.sync_threads()")
    probe(e, "a.kvload")
    e("# scores: thread t owns position t % 128 and half of this head's dimensions")
    e("acc[0] = 0.0", "fv[0] = 0.0", "fv[1] = 0.0", "fv[2] = 0.0")
    e.block(f"for _d in T.serial({DH // 2 // 4}):")
    for j, slot in enumerate(("acc[0]", "fv[0]", "fv[1]", "fv[2]")):
        e(f"{slot} += fs[bx * {DH} + (tid // {ABLK}) * {DH // 2} + _d * 4 + {j}]"
          f" * T.Cast(\"float32\", ks[tid % {ABLK},"
          f" (tid // {ABLK}) * {DH // 2} + _d * 4 + {j}])")
    e.end()
    e("red[tid] = acc[0] + fv[0] + fv[1] + fv[2]", "T.sync_threads()")
    e.block(f"if tid < {ABLK}:")
    e(f"fs[{QP} + tid] = T.if_then_else(tid < iv[0],"
      f" bf(red[tid] + red[{ABLK} + tid]), NEG)").end()
    e("T.sync_threads()")
    probe(e, "a.scores")
    e("# this head's 128 scores: a quarter each to 32 threads, then one thread")
    e.block("if tid < 32:")
    e(f"red[tid] = T.max(T.max(fs[{QP} + tid], fs[{QP} + 32 + tid]),"
      f" T.max(fs[{QP} + 64 + tid], fs[{QP} + 96 + tid]))").end()
    e("T.sync_threads()")
    e.block("if tid == 0:")
    e("acc[0] = NEG")
    e.block("for _u in T.unroll(32):")("acc[0] = T.max(acc[0], red[_u])").end()
    e(f"fv[0] = T.max(sml[{4 * HQ + 1}], acc[0])",
      f"sml[{4 * HQ + 3}] = T.exp(sml[{4 * HQ + 1}] - fv[0])",
      f"sml[{4 * HQ + 1}] = fv[0]").end()
    e("T.sync_threads()")
    e.block(f"if tid < {ABLK}:")
    e(f"fs[{QP} + tid] = T.exp(fs[{QP} + tid] - sml[{4 * HQ + 1}])",
      f"red[tid] = fs[{QP} + tid]").end()
    e("T.sync_threads()")
    e.block("if tid == 0:")
    e("acc[0] = 0.0")
    e.block(f"for _u in T.unroll({ABLK}):")("acc[0] += red[_u]").end()
    e(f"sml[{4 * HQ + 2}] = sml[{4 * HQ + 2}] * sml[{4 * HQ + 3}] + acc[0]").end()
    e("T.sync_threads()")
    probe(e, "a.softmax")
    e("# accumulate: thread t owns dimension t % 128 and half of the positions")
    e(f"av[0] = av[0] * sml[{4 * HQ + 3}]")
    e.block(f"for _p in T.unroll({ABLK // 2}):")
    e.block(f"if (tid // {ABLK}) * {ABLK // 2} + _p < iv[0]:")
    e(f"av[0] += fs[{QP} + (tid // {ABLK}) * {ABLK // 2} + _p]"
      f" * T.Cast(\"float32\", vs[(tid // {ABLK}) * {ABLK // 2} + _p, tid % {ABLK}])")
    e.end().end()
    e("T.sync_threads()")
    probe(e, "a.acc")
    e.end()
    e("# this worker's partial for this head, for the four-way merge")
    e("red[tid] = av[0]", "T.sync_threads()")
    e.block(f"if tid < {DH}:")
    e(f"scratch[S_ATT + (by * NHEAD + bx) * {DH + 2} + tid] = red[tid] + red[{ABLK} + tid]")
    e.end()
    e.block("if tid == 0:")
    e(f"scratch[S_ATT + (by * NHEAD + bx) * {DH + 2} + {DH}] = sml[{4 * HQ + 1}]",
      f"scratch[S_ATT + (by * NHEAD + bx) * {DH + 2} + {DH + 1}] = sml[{4 * HQ + 2}]").end()
    probe(e, "a.accum")
    e("T.sync_grid()")
    probe(e, "a.bar")
    e("# merge the four workers of one head; only the first row of the grid does it")
    e.block("if by == 0:")
    e.block("if tid == 0:")
    e("acc[0] = NEG")
    e.block("for _c in T.serial(NW):")
    e(f"acc[0] = T.max(acc[0], scratch[S_ATT + (_c * NHEAD + bx) * {DH + 2} + {DH}])").end()
    e("sml[MAXSLOT] = acc[0]").end()
    e("T.sync_threads()")
    e.block(f"if tid < {DH + 1}:")
    e("acc[0] = 0.0")
    e.block("for _c in T.serial(NW):")
    e(f"acc[0] += T.exp(scratch[S_ATT + (_c * NHEAD + bx) * {DH + 2} + {DH}]"
      f" - sml[MAXSLOT]) * scratch[S_ATT + (_c * NHEAD + bx) * {DH + 2}"
      f" + T.if_then_else(tid < {DH}, tid, {DH + 1})]").end()
    e("red[tid] = acc[0]").end()
    e("T.sync_threads()")
    e.block(f"if tid < {DH}:")
    e(f"scratch[S_CTX + bx * {DH} + tid] = bf(red[tid] / red[{DH}])").end().end()


def attn_by_context(e: Emit):
    """A stripe of the context per CTA, on the tensor cores, merged by log-sum-exp.

    Past a few thousand positions this is the only split with enough units in
    it: 32 heads is 32 CTAs however long the context gets, while blocks of it
    are `ceil(T/128)` and fill the grid.

    Both products of a block are tensor-core GEMMs, one pair per KV group:

        scores   Q(16 heads x 128 dims) @ K^T(128 dims x 128 positions)
        context  P(16 heads x 128 positions) @ V(128 positions x 128 dims)

    M is 16 because GQA puts 16 query heads on one KV head and a decode step has
    one query token, so `T.gemm` picks `mma.m16n8k16` rather than the warpgroup
    form, whose M is fixed at 64 and would run three quarters empty. What that
    buys is not the peak: a block's 128 KB of K and V is microseconds of this
    CTA's share of HBM against 154 ns of tensor core, so the products stop being
    what the loop waits for at all.

    Everything the tensor cores touch stays in the layout they chose. The scores
    land in a shared bf16 block, the softmax runs over that block in place, and
    the same block is the second product's left operand -- no f32 staging, and
    nothing crosses between a fragment layout and a linear one, which is the
    one shape of copy that costs more than the products do.
    """
    e("# ---- a stripe of the context per CTA, both KV heads, ABLK at a time",
      f"_tot = T.ceildiv(ctx_all, {ABLK})",
      f"_cb = T.ceildiv(_tot, {CTAS})",
      "_b0 = cta * _cb",
      "_nblk = T.max(T.min(_cb, _tot - _b0), 0)")
    e("# Q is already scaled in `fs`, one head to a row; the tensor cores want it",
      "# in bf16, and it fits in registers.")
    e(f"T.copy(fs2[0:{GQA}, :], qf0)", f"T.copy(fs2[{GQA}:{2 * GQA}, :], qf1)")
    e.block(f"if tid < {HQ}:")("sml[tid] = NEG", f"sml[{HQ} + tid] = 0.0").end()
    e("T.clear(of0)", "T.clear(of1)", "T.sync_threads()")
    e.block("for _b in T.serial(_nblk):")
    e("_bi = _b0 + _b",
      f"iv[0] = T.max(T.min(ctx_all - _bi * {ABLK}, {ABLK}), 0)")
    for g, (qf, of) in enumerate((("qf0", "of0"), ("qf1", "of1"))):
        e(f"# ---- KV group {g}")
        kv_load(e, "k", "ks", str(g * DH), f"k{g}")
        kv_load(e, "v", "vs", str(g * DH), f"v{g}")
        e("T.sync_threads()")
        probe(e, "a.kvload")
        e("T.clear(sfg)")
        e(f"T.gemm({qf}, ks, sfg, transpose_B=True)")
        e("T.copy(sfg, ps)", "T.sync_threads()")
        probe(e, "a.scores")
        e("# The online softmax, in place over the score block: a head to sixteen",
          "# threads and eight positions each. Those sixteen are sixteen adjacent",
          "# lanes of one warp, so both reductions are butterflies over registers",
          "# -- no shared round trip, no barrier, and the 240 threads a sixteen-",
          "# wide serial fold would have left idle keep their own heads.")
        e("acc[0] = NEG")
        e.block("for _u in T.unroll(8):")
        e.block("if (tid % 16) * 8 + _u < iv[0]:")
        e("acc[0] = T.max(acc[0], T.Cast(\"float32\","
          " ps[tid // 16, (tid % 16) * 8 + _u]))")
        e.end().end()
        e.block("for _d in T.unroll(4):")
        e("acc[0] = T.max(acc[0], T.shfl_xor(acc[0], T.shift_left(1, _d)))").end()
        e(f"fv[0] = T.max(sml[{g * GQA} + tid // 16], acc[0])",
          f"fv[2] = T.exp(sml[{g * GQA} + tid // 16] - fv[0])")
        e("# a position past the end of the context gets a zero, which keeps it",
          "# out of the sum and out of the product that follows")
        e("acc[0] = 0.0")
        e.block("for _u in T.unroll(8):")
        e("fv[1] = T.if_then_else((tid % 16) * 8 + _u < iv[0],"
          " T.exp(T.Cast(\"float32\", ps[tid // 16, (tid % 16) * 8 + _u])"
          " - fv[0]), 0.0)")
        e("ps[tid // 16, (tid % 16) * 8 + _u] = bf(fv[1])")
        e("acc[0] += fv[1]").end()
        e.block("for _d in T.unroll(4):")
        e("acc[0] += T.shfl_xor(acc[0], T.shift_left(1, _d))").end()
        e.block("if tid % 16 == 0:")
        e(f"sml[{HQ + g * GQA} + tid // 16] ="
          f" sml[{HQ + g * GQA} + tid // 16] * fv[2] + acc[0]",
          f"sml[{g * GQA} + tid // 16] = fv[0]",
          f"sml[{3 * HQ + g * GQA} + tid // 16] = fv[2]").end()
        e("T.sync_threads()")
        probe(e, "a.softmax")
        e("# rescale what the earlier blocks left, then add this block's share")
        e.block(f"for _i, _j in T.Parallel({GQA}, {DH}):")
        e(f"{of}[_i, _j] = {of}[_i, _j] * sml[{3 * HQ + g * GQA} + _i]").end()
        e(f"T.gemm(ps, vs, {of})")
        probe(e, "a.acc")
    e.end()
    e("# the stripe's state: 32 context vectors and their (max, sum)")
    e(f"T.copy(of0, fs2[0:{GQA}, :])", f"T.copy(of1, fs2[{GQA}:{2 * GQA}, :])")
    e("T.sync_threads()")
    spread(e, QP, [f"scratch[S_ATT + (cta * {HQ} + _j // {DH}) * {DH + 2}"
                   f" + _j % {DH}] = fs[_j]"])
    e.block(f"if tid < {HQ}:")
    e(f"scratch[S_ATT + (cta * {HQ} + tid) * {DH + 2} + {DH}] = sml[tid]",
      f"scratch[S_ATT + (cta * {HQ} + tid) * {DH + 2} + {DH + 1}] = sml[{HQ} + tid]").end()
    probe(e, "a.accum")
    e("T.sync_grid()")
    probe(e, "a.bar")
    e("# merge the stripes: one query head per CTA, over the stripes that exist")
    e(f"_nact = T.max(T.min(T.ceildiv(_tot, _cb), {CTAS}), 1)")
    e.block(f"if cta < {HQ}:")
    e.block("if tid == 0:")("acc[0] = NEG")
    e.block("for _c in T.serial(_nact):")
    e(f"acc[0] = T.max(acc[0], scratch[S_ATT + (_c * {HQ} + cta) * {DH + 2} + {DH}])").end()
    e("sml[MAXSLOT] = acc[0]").end()
    e("T.sync_threads()")
    e.block(f"if tid < {DH + 1}:")
    e("acc[0] = 0.0")
    e.block("for _c in T.serial(_nact):")
    e(f"acc[0] += T.exp(scratch[S_ATT + (_c * {HQ} + cta) * {DH + 2} + {DH}]"
      f" - sml[MAXSLOT]) * scratch[S_ATT + (_c * {HQ} + cta) * {DH + 2}"
      f" + T.if_then_else(tid < {DH}, tid, {DH + 1})]").end()
    e("red[tid] = acc[0]").end()
    e("T.sync_threads()")
    e.block(f"if tid < {DH}:")
    e(f"scratch[S_CTX + cta * {DH} + tid] = bf(red[tid] / red[{DH}])").end().end()


def attention(e: Emit):
    """One decode step of full attention, by whichever split has the units in it.

    Both splits leave the same thing behind -- head `h`'s context vector at
    `S_CTX + h*DH` -- so the projection that follows does not know which ran.
    The choice is grid-uniform (`ctx_all` comes from the address table, the same
    for every CTA), which is what makes it safe for the two arms to hold
    different numbers of `sync_grid`.

    Per-CTA work is `nblk` blocks x 1 head for the head split and
    `ceil(nblk/132)` blocks x 32 heads for the context split, so they cross at
    nblk = 32. `HSPLIT` is where the emitted branch puts the crossing.
    """
    e("# ============================== attention")
    # The whole-block view ends where the remainder begins, and both are counted
    # in blocks: `ctx_full` is a multiple of ABLK by construction, so a block is
    # wholly in one or wholly in the other.
    e("k_tail = T.make_tensor_from_addr(ptrs[P_KT + at], (1 << 16,), \"bfloat16\")",
      "v_tail = T.make_tensor_from_addr(ptrs[P_VT + at], (1 << 16,), \"bfloat16\")")
    e(f"_cfb = T.ceildiv(ctx_full, {ABLK})")
    probe(e, "prenorm")
    gemv(e, "wqkv", NUM["wqkv"], "at", 12386304, QP + 2 * KVP, H, 0,
         "qkv", "S_QKV")
    probe(e, "a.qkv")
    e("T.sync_grid()")
    probe(e, "a.bar")
    spread(e, QP, ["fs[_j] = scratch[S_QKV + _j] * QSCALE"])
    e("T.sync_threads()")
    only = __import__("os").environ.get("NEMO_ONLY")
    if only == "ctx":
        attn_by_context(e)
    elif only == "head":
        attn_by_head(e)
    else:
        e.block(f"if ctx_full < {HSPLIT}:")
        attn_by_head(e)
        e.end()
        e.block("else:")
        attn_by_context(e)
        e.end()
    probe(e, "a.combine")
    e("T.sync_grid()")
    probe(e, "a.bar")
    copy_to_fs(e, "S_CTX", QP)
    gemv(e, "wo", NUM["wo"], "at", 11010048, H, QP, 0, "bf", "S_MIX")
    probe(e, "a.out_proj")
    e("T.sync_grid()")
    probe(e, "a.bar")
    residual(e)
    probe(e, "residual")


def moe(e: Emit):
    e("# ============================== MoE")
    probe(e, "prenorm")
    gemv(e, "wrt", NUM["wrt"], "at", 344064, E, H, 0, "f32", "S_RLOG")
    probe(e, "e.router")
    e("T.sync_grid()")
    probe(e, "e.bar")
    e("# every CTA picks the same six experts from the same logits: 128 values",
      "# is not work worth a barrier to share. One warp does it: each lane keeps",
      "# the best of four experts, a butterfly picks the best of the warp, and the",
      "# winner is struck out of the scores before the next round. Ties go to the",
      "# lower index, which is what `topk` does.")
    e.block(f"if tid < {E}:")
    e(f"fs[F_SCORE + tid] = 1.0 / (1.0 + T.exp(0.0 - scratch[S_RLOG + tid]))"
      " + eb[at * E + tid]").end()
    e("T.sync_threads()")
    e.block("if tid < 32:")
    e.block(f"for _j in T.serial({KTOP}):")
    e("fv[0] = NEG", "iv[0] = 0")
    e.block(f"for _q in T.serial({E // 32}):")
    e.block("if fs[F_SCORE + _q * 32 + lane] > fv[0]:")
    e("fv[0] = fs[F_SCORE + _q * 32 + lane]", "iv[0] = _q * 32 + lane").end().end()
    e.block("for _d in T.unroll(5):")
    e("fv[1] = T.shfl_xor(fv[0], T.shift_left(1, _d))",
      "iv[1] = T.shfl_xor(iv[0], T.shift_left(1, _d))")
    e.block("if fv[1] > fv[0]:")("fv[0] = fv[1]", "iv[0] = iv[1]").end()
    e.block("else:")
    e.block("if fv[1] == fv[0]:")
    e.block("if iv[1] < iv[0]:")("iv[0] = iv[1]").end().end().end().end()
    e.block("if lane == 0:")
    e("sel[_j] = T.Cast(\"float32\", iv[0])",
      f"sel[{KTOP} + _j] = 1.0 / (1.0 + T.exp(0.0 - scratch[S_RLOG + iv[0]]))",
      "fs[F_SCORE + iv[0]] = NEG").end()
    e("T.sync_warp()").end()
    e.block("if lane == 0:")
    e("acc[0] = 0.0")
    e.block(f"for _j in T.serial({KTOP}):")(f"acc[0] += sel[{KTOP} + _j]").end()
    e.block(f"for _j in T.serial({KTOP}):")
    e(f"sel[{KTOP} + _j] = sel[{KTOP} + _j] / (acc[0] + 1e-20) * RSCALE").end().end().end()
    e("T.sync_threads()")
    probe(e, "e.topk")
    # The six routed experts are one pass over twelve tiles, not six passes over
    # two. A two-tile pass pays a cold memory pipeline for both of its tiles, and
    # six copies of the body are six times the instruction bytes to fetch.
    gemv(e, "wup", NUM["wup"],
         lambda u: f"at * E + T.Cast(\"int32\", sel[{u}])", 4988928,
         I, H, 0, "relu2", lambda u: f"S_MID + ({u}) * {I}", nunit=KTOP)
    probe(e, "e.up")
    gemv(e, "wsu", NUM["wsu"], "at", 9977856, IS, H, 0, "relu2",
         f"S_MID + {KTOP * I}")
    probe(e, "e.sh_up")
    e("T.sync_grid()")
    probe(e, "e.bar")
    nrh = (H + CTAS - 1) // CTAS
    # The layer's twenty-one output rows are accumulated in this CTA's own shared
    # slice and published once, so the seven experts do not each pay a global
    # read-modify-write per tile on the way to the same row.
    e.block(f"if tid < {nrh}:")("fs[F_MIX + tid] = 0.0").end()
    e("T.sync_threads()")
    e("# the six mid vectors go to the tail of the arena, so the down half is one",
      "# pass too and its x is a shared read like any other. They are already bf16",
      "# values -- the up half rounded them when it squared them -- so nothing is",
      "# lost by holding them in the arena's own type.")
    spread(e, KTOP * I,
           ["sm[MIDOFF + _j] = T.Cast(\"bfloat16\", scratch[S_MID + _j])"])
    e("T.sync_threads()")
    gemv(e, "wdn", NUM["wdn"],
         lambda u: f"at * E + T.Cast(\"int32\", sel[{u}])", 4988928,
         H, I, lambda u: f"MIDOFF + ({u}) * {I}", "local", "F_MIX",
         nunit=KTOP, arena=MIDOFF_VAL, xsrc="sm",
         scale=lambda u: f"sel[{KTOP} + ({u})]")
    probe(e, "e.down")
    copy_to_fs(e, f"S_MID + {KTOP * I}", IS)
    gemv(e, "wsd", NUM["wsd"], "at", 9977856, H, IS, 0, "localraw", "F_SH")
    e.block(f"if tid < {nrh}:")
    e(f"_r = cta * {nrh} + tid")
    e.block("if _r < H:")
    e("scratch[S_MIX + _r] = bf(bf(fs[F_MIX + tid]) + bf(fs[F_SH + tid]))")
    e.end().end()
    probe(e, "e.sh_down")
    e("T.sync_grid()")
    probe(e, "e.bar")
    residual(e)
    probe(e, "residual")


def body() -> str:
    e = Emit(indent=3)
    e(f"cta = bx + by * {NHEAD}",
      "tid = T.get_thread_binding()", "warp = tid // 32", "lane = tid % 32",
      "ctx_full = T.Cast(\"int32\", ptrs[NPTR + 0])",
      "ctx_tail = T.Cast(\"int32\", ptrs[NPTR + 1])",
      "cur_pos = T.Cast(\"int32\", ptrs[NPTR + 2])",
      "ctx_all = ctx_full + ctx_tail",
      "")
    e("# ---- embedding: this token's own row, read by every CTA. No barrier",
      "# follows it -- every CTA built the same row out of the same table.",
      "_tok = T.Cast(\"int32\", token[0])")
    e("acc[0] = 0.0")
    spread(e, H, ["hs[_j] = table[_tok * H + _j]",
                  "acc[0] += T.Cast(\"float32\", hs[_j])"
                  " * T.Cast(\"float32\", hs[_j])"])
    cta_sum(e, "acc[0]", "sml[SUMSLOT]")
    probe(e, "embed")
    e("")
    e("nlayer = T.Cast(\"int32\", ptrs[NPTR + 3])")
    e.block("for layer in T.serial(nlayer):")
    # Both indices are clamped where they are read, not where they are used.
    # The layer counter's bound is a kernel argument, so nothing downstream can
    # prove a table read stays inside its table, and every such read comes out
    # as a branch around the load -- which is a branch the loads after it cannot
    # be issued past.
    e(f"lay = T.max(T.min(layer, {NL - 1}), 0)")
    e("kid = lmeta[lay * 2]",
      f"at = T.max(T.min(lmeta[lay * 2 + 1], {max(N_MAMBA, N_ATTN, N_MOE) - 1}), 0)")
    rmsnorm(e, "T.Cast(\"float32\", gam[lay * H + _j])")
    e.block("if kid == 0:")
    mamba(e)
    e.end()
    e.block("else:")
    e.block("if kid == 1:")
    attention(e)
    e.end()
    e.block("else:")
    moe(e)
    e.end().end().end()
    e("")
    e("# ---- the closing norm and the head")
    rmsnorm(e, "T.Cast(\"float32\", gf[_j])")
    probe(e, "final_norm")
    gemv(e, "whead", NUM["whead"], "0", 0, V, H, 0, "logits")
    probe(e, "head")
    return e.text()



def attn_body(arm: str) -> str:
    """The attention stage on its own, for the twin's `attend*` functions.

    The step inlines this -- a Call from one HIR Function to another is a device
    call inside the same kernel invocation -- so what stands here is the same two
    bodies, emitted from the same code, over the same buffers. What differs is
    only where the ends are: Q arrives in a parameter instead of in the step's
    own scratch, and the context vectors leave in one instead of being read by
    the projection that would have followed.
    """
    global PROF
    was, PROF = PROF, False
    e = Emit(indent=4)
    e(f"cta = bx + by * {NHEAD}",
      "tid = T.get_thread_binding()", "warp = tid // 32", "lane = tid % 32",
      "at = 0",
      "ctx_full = meta[0]", "ctx_tail = meta[1]", "ctx_all = ctx_full + ctx_tail",
      f"_cfb = T.ceildiv(ctx_full, {ABLK})")
    spread(e, QP, [f"fs[_j] = T.Cast(\"float32\", qg[_j]) * QSCALE"])
    e("T.sync_threads()")
    if arm == "head":
        attn_by_head(e)
    else:
        attn_by_context(e)
    e("T.sync_grid()")
    spread(e, QP, ["out[_j] = bf(scratch[S_CTX + _j])"])
    PROF = was
    return e.text()


ATTN = '''

# ---------------------------------------------------------------------------
# The same attention, on its own: what the twin answers `attend*` with.
# ---------------------------------------------------------------------------

@tilelang.jit
def build_attn(arm):
    @T.prim_func
    def attn(
        qg: T.Tensor((QP,), "bfloat16"),
        ptrs: T.Tensor((NPTR,), "int64"),
        meta: T.Tensor((2,), "int32"),
        scratch: T.Tensor((SCRATCH,), "float32"),
        out: T.Tensor((QP,), "bfloat16"),
    ):
        with T.Kernel(NHEAD, NW, threads=THREADS) as (bx, by):
            ks = T.alloc_shared((ABLK, DH), "bfloat16")
            vs = T.alloc_shared((ABLK, DH), "bfloat16")
            ps = T.alloc_shared((GQA, ABLK), "bfloat16")
            fs2 = T.alloc_shared((FROW, DH), "float32")
            T.annotate_layout({{fs2: T.Layout((FROW, DH), lambda i, j: i * DH + j)}})
            fs = T.view(fs2, (FARENA,))
            red = T.alloc_shared((THREADS,), "float32")
            sml = T.alloc_shared((4 * HQ + 4,), "float32")
            acc = T.alloc_local((1,), "float32")
            av = T.alloc_local((16,), "float32")
            fv = T.alloc_local((4,), "float32")
            iv = T.alloc_local((4,), "int32")
            of0 = T.alloc_fragment((GQA, DH), "float32")
            of1 = T.alloc_fragment((GQA, DH), "float32")
            sfg = T.alloc_fragment((GQA, ABLK), "float32")
            qf0 = T.alloc_fragment((GQA, DH), "bfloat16")
            qf1 = T.alloc_fragment((GQA, DH), "bfloat16")
            if arm == "head":
{head_body}
            else:
{ctx_body}
    return attn


class AttnRunner:
    """What the standalone attention keeps between calls."""

    _cache = {{}}

    @classmethod
    def get(cls, arm, device):
        key = (arm, str(device))
        if key not in cls._cache:
            cls._cache[key] = (build_attn(arm),
                               torch.zeros(SCRATCH, dtype=torch.float32, device=device),
                               torch.zeros(NPTR, dtype=torch.int64, device=device),
                               torch.zeros(2, dtype=torch.int32, device=device),
                               torch.zeros(QP, dtype=torch.bfloat16, device=device))
        return cls._cache[key]


def run_attn(arm, qg, k_cache, v_cache, k_tail, v_tail):
    """One decode step of full attention, by the named placement.

    `arm` is "head" or "context"; "dispatch" reads the crossover the semantics
    states off the whole-block context length, which is the number the mega
    step's own branch is on.
    """
    ctx_full, ctx_tail = k_cache.shape[1], k_tail.shape[1]
    if arm == "dispatch":
        arm = "head" if ctx_full < HSPLIT else "context"
    kernel, scratch, ptrs, meta, out = AttnRunner.get(arm, qg.device)
    ptrs[P_KC] = k_cache.data_ptr()
    ptrs[P_VC] = v_cache.data_ptr()
    ptrs[P_KT] = k_tail.data_ptr()
    ptrs[P_VT] = v_tail.data_ptr()
    meta[0], meta[1] = ctx_full, ctx_tail
    kernel(qg.reshape(-1), ptrs, meta, scratch, out)
    return out.reshape(1, QP)
'''

HEADER = '''"""One launch: the whole decode step as a single persistent cooperative kernel.

Generated by ``gen_kernel.py``; edit the generator, not this file. TileLang
rewrites every `for` in a `@T.prim_func` into a device loop and does not bind a
nested `def`, so trace-time repetition -- a prefetch prologue, sixteen
accumulator registers, one body per stage kind -- has to be written out.

The shape is the one `model.py` states. Every stage of the authored program is a
stage here, and where the HIR reshards a value back out of its mesh split this
kernel calls `T.sync_grid()`. Nothing returns between stages: the grid is
resident for the whole step and a barrier is what ends a stage.

    layer 0..51
        mamba      in_proj |sync| conv + SSM scan |sync| gated norm + out_proj |sync|
        attention  qkv     |sync| KV scan         |sync| combine |sync| o_proj |sync|
        moe        router  |sync| up x7           |sync| down x7 + mix         |sync|
    closing norm + lm_head

The residual add and the next layer's pre-norm are done by every CTA from the
published hidden row, so they need no barrier of their own: 2688 elements is not
work worth dividing, and dividing it would cost a barrier to put back together.
The same reasoning puts the router's 128 logits and the Mamba B/C convolution in
every CTA rather than behind a barrier.

**How a stage moves its weights.** Every stage is a matrix-vector product whose
matrix is stored `(out, in)`. A CTA owns a run of output rows, so its slice is a
contiguous run of bytes, and it streams that run through shared memory eight
rows at a time: `T.tma_copy` issues the bulk load, one elected lane arrives on
that tile's mbarrier, and `T.mbarrier_wait_parity` is where the consumer waits.
The loads run `NS` tiles ahead, so the tiles a stage issues last are still in
flight when it reaches its `T.sync_grid()`.

**How attention moves its K and V.** Not that way. Those addresses come out of
the table, and a TensorMap descriptor is encoded on the host, so a bulk copy from
one of them is the rank-1 kind that writes physically linear bytes -- and a
buffer pinned linear costs `T.gemm` six times its own operand loads. A plain
`T.copy` writes through whatever layout the tensor cores picked and reaches the
same 4.15 TB/s (`kbench/blkloop.py`), so attention's two operand buffers are
filled by `T.copy(..., prefer_instruction="cp_async")` and the products are
`mma.m16n8k16` on them.

Each stage leaves every mbarrier on an even completion count -- adding one empty
arrive/wait where the tile count made it odd -- so a stage's parity is
`(tile // NS) % 2` and no phase has to be carried across the layer loop.

**Why the weights are packed.** A kernel cannot take 401 pointers. `load` copies
every weight into one flat tensor per weight class with the layer on the leading
axis, so the kernel takes eighteen buffers and reaches any weight by (layer,
row). State and K/V caches belong to the caller and change identity between
steps, so they arrive as a table of addresses and the kernel opens each with
`T.make_tensor_from_addr`.
"""
from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

import model

H, V, EPS = model.H, model.V, model.EPS
MH, MD, SS, NG, MI, CONV, KER, WIN, PROJ = (model.MH, model.MD, model.SS, model.NG,
                                            model.MI, model.CONV, model.KER, model.WIN,
                                            model.PROJ)
GRP, HPG, DTMIN = model.GRP, model.HPG, model.DTMIN
HQ, HKV, DH, QP, KVP, GQA, QSCALE = (model.HQ, model.HKV, model.DH, model.QP,
                                     model.KVP, model.GQA, model.QSCALE)
E, KTOP, I, IS, RSCALE = model.E, model.K, model.I, model.IS, model.RSCALE
KINDS = model.LAYER_KINDS
NL = len(KINDS)

NHEAD = {NHEAD}
NW = {NW}
CTAS = {CTAS}
THREADS = {THREADS}
RB = {RB}
#: Where the two attention placements change over; the same number the
#: authored HIR dispatches on (`attention.py`'s CROSSOVER).
HSPLIT = {HSPLIT}
ARENA = {ARENA}
FROW = {FROW}
FARENA = {FARENA}
ABLK = {ABLK}
NEG = -3.0e38
KVROW = HKV * DH
#: How many ABLK blocks the K/V views declare. The cache holds 262144 positions
#: at most, which is 2048 of them; the remainder is two.
CBLK = 4096

_KIND_ID = {{"linear_attention": 0, "full_attention": 1, "moe": 2}}
_LAYER_META, _seen = [], {{0: 0, 1: 0, 2: 0}}
for _k in KINDS:
    _LAYER_META.append((_KIND_ID[_k], _seen[_KIND_ID[_k]]))
    _seen[_KIND_ID[_k]] += 1
N_MAMBA, N_ATTN, N_MOE = _seen[0], _seen[1], _seen[2]

P_CONV_IN, P_SSM_IN = 0, N_MAMBA
P_CONV_OUT, P_SSM_OUT = 2 * N_MAMBA, 3 * N_MAMBA
P_KC = 4 * N_MAMBA
P_VC, P_KT, P_VT = P_KC + N_ATTN, P_KC + 2 * N_ATTN, P_KC + 3 * N_ATTN
NPTR = 4 * N_MAMBA + 4 * N_ATTN
#: Three more slots carry ctx_full, ctx_tail and cur_pos, so one small
#: host-to-device copy per step hands the kernel everything that changes.
NTAB = NPTR + 5

# gmem scratch, in f32 elements. All of it is small enough to stay in L2.
S_H = 0
#: Two hidden rows, alternating per layer; see `residual`.
S_PROJ = S_H + 2 * H
S_Y = S_PROJ + PROJ
S_MIX = S_Y + MI
S_TMP = S_MIX + MI
S_QKV = S_TMP + H
S_CTX = S_QKV + QP + 2 * KVP
S_RLOG = S_CTX + QP
S_MID = S_RLOG + E
S_ATT = S_MID + KTOP * I + IS
SCRATCH = S_ATT + CTAS * HQ * (DH + 2)

#: fs (shared f32) while a Mamba layer's scan runs.
MIDOFF = ARENA - KTOP * I
F_BC = 0
F_XV = F_BC + 2 * NG * SS
#: The MoE's own f32 slots, below the QP an attention layer stages Q into: the
#: two kinds of layer never hold both, and the f32 arena is 33 rows because of
#: what is above them. Each has to clear the longest vector the stage that
#: follows it reads out of the same arena -- H for the router's scores, IS for
#: the two the shared expert leaves behind.
F_SCORE = H
F_MIX = IS
F_SH = F_MIX + (H + CTAS - 1) // CTAS + 1
#: Scalar slots in the small shared array.
SUMSLOT = 4 * HQ
MAXSLOT = 4 * HQ + 1

PACK = {{
    "win": (N_MAMBA, PROJ, H), "wout": (N_MAMBA, H, MI),
    "convw": (N_MAMBA, CONV, KER), "convb": (N_MAMBA, CONV), "ggdn": (N_MAMBA, MI),
    "wqkv": (N_ATTN, QP + 2 * KVP, H), "wo": (N_ATTN, H, QP),
    "wrt": (N_MOE, E, H), "wup": (N_MOE, E, I, H), "wdn": (N_MOE, E, H, I),
    "wsu": (N_MOE, IS, H), "wsd": (N_MOE, H, IS),
    "gam": (NL, H), "table": (V, H), "whead": (V, H),
    "mscal": (N_MAMBA, 3, MH), "gf": (H,),
}}
PACK_F32 = {{"eb": (N_MOE, E)}}
PACK_ORDER = list(PACK) + list(PACK_F32)
RING = 4


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def bf(x):
    """Round to bf16 and go on in f32, the way a bf16 op does."""
    return T.Cast("float32", T.Cast("bfloat16", x))


class Packed:
    """Every weight of one class in one flat tensor."""

    def __init__(self, device):
        self.shape = dict(PACK) | dict(PACK_F32)
        self.t = {{n: torch.zeros(_numel(s), dtype=torch.bfloat16, device=device)
                  for n, s in PACK.items()}}
        self.t.update({{n: torch.zeros(_numel(s), dtype=torch.float32, device=device)
                       for n, s in PACK_F32.items()}})

    def view(self, name):
        return self.t[name].view(self.shape[name])

    def flat_order(self):
        return [self.t[n] for n in PACK_ORDER]


def pack_into(packed: Packed, name: str, value: torch.Tensor):
    """Copy one declared weight into its slot; return the view standing for it."""
    def put(key, *idx):
        dst = packed.view(key)
        for j in idx:
            dst = dst[j]
        dst.copy_(value.reshape(dst.shape).to(dst.dtype))
        return dst

    if name in ("table", "w_head"):
        return put({{"table": "table", "w_head": "whead"}}[name])
    if name == "gamma_final":
        return put("gf")
    layer = int(name.split("_")[0][1:])
    kid, at = _LAYER_META[layer]
    tail = name[name.index("_") + 1:]
    if tail == "gamma":
        return put("gam", layer)
    if kid == 0:
        simple = {{"w_in": "win", "w_out": "wout", "conv_w": "convw",
                  "conv_b": "convb", "gamma_gdn": "ggdn"}}
        if tail in simple:
            return put(simple[tail], at)
        return put("mscal", at, {{"a_log": 0, "dt_bias": 1, "d_skip": 2}}[tail])
    if kid == 1:
        if tail == "w_o":
            return put("wo", at)
        off = {{"w_q": 0, "w_k": QP, "w_v": QP + KVP}}[tail]
        dst = packed.view("wqkv")[at, off:off + value.shape[0]]
        dst.copy_(value)
        return dst
    simple = {{"w_router": "wrt", "w_up": "wup", "w_down": "wdn",
              "w_sh_up": "wsu", "w_sh_down": "wsd", "e_bias": "eb"}}
    return put(simple[tail], at)


@tilelang.jit
def build():
    @T.prim_func
    def mega(
        win: T.Tensor((_numel(PACK["win"]),), "bfloat16"),
        wout: T.Tensor((_numel(PACK["wout"]),), "bfloat16"),
        convw: T.Tensor((_numel(PACK["convw"]),), "bfloat16"),
        convb: T.Tensor((_numel(PACK["convb"]),), "bfloat16"),
        ggdn: T.Tensor((_numel(PACK["ggdn"]),), "bfloat16"),
        wqkv: T.Tensor((_numel(PACK["wqkv"]),), "bfloat16"),
        wo: T.Tensor((_numel(PACK["wo"]),), "bfloat16"),
        wrt: T.Tensor((_numel(PACK["wrt"]),), "bfloat16"),
        wup: T.Tensor((_numel(PACK["wup"]),), "bfloat16"),
        wdn: T.Tensor((_numel(PACK["wdn"]),), "bfloat16"),
        wsu: T.Tensor((_numel(PACK["wsu"]),), "bfloat16"),
        wsd: T.Tensor((_numel(PACK["wsd"]),), "bfloat16"),
        gam: T.Tensor((_numel(PACK["gam"]),), "bfloat16"),
        table: T.Tensor((_numel(PACK["table"]),), "bfloat16"),
        whead: T.Tensor((_numel(PACK["whead"]),), "bfloat16"),
        mscal: T.Tensor((_numel(PACK["mscal"]),), "bfloat16"),
        gf: T.Tensor((H,), "bfloat16"),
        eb: T.Tensor((_numel(PACK_F32["eb"]),), "float32"),
        ptrs: T.Tensor((NTAB,), "int64"),
        lmeta: T.Tensor((NL * 2,), "int32"),
        token: T.Tensor((1,), "int64"),
        scratch: T.Tensor((SCRATCH,), "float32"),
        logits: T.Tensor((V,), "float32"),
    ):
        with T.Kernel(NHEAD, NW, threads=THREADS) as (bx, by):
            sm = T.alloc_shared((ARENA,), "bfloat16")
            # Attention's operands, in the layout `T.gemm` chooses. A plain copy
            # writes through that layout, which is why K and V come in by
            # `T.copy` rather than by the bulk path the streaming stages use --
            # measured at the same 4.13 TB/s, and worth six times its own cost
            # in what the tensor cores then do with the bytes. One KV group
            # wide, which is what fits and what either arm reads.
            ks = T.alloc_shared((ABLK, DH), "bfloat16")
            vs = T.alloc_shared((ABLK, DH), "bfloat16")
            # The block's scores, then its probabilities in place: written by a
            # gemm, read and rewritten by the softmax, read by the next gemm.
            ps = T.alloc_shared((GQA, ABLK), "bfloat16")
            # The f32 scratch is two-dimensional for the same reason -- a gemm's
            # accumulator lands in it -- and pinned linear so the flat view every
            # other stage indexes it through means what it says.
            fs2 = T.alloc_shared((FROW, DH), "float32")
            T.annotate_layout({{fs2: T.Layout((FROW, DH), lambda i, j: i * DH + j)}})
            fs = T.view(fs2, (FARENA,))
            # The hidden row, one copy a CTA for the whole step. bf16, because
            # every value that lands in it was rounded there first -- the
            # embedding table is bf16 and the residual publishes `bf(...)` --
            # so the f32 it used to be held five kilobytes of zeros.
            hs = T.alloc_shared((H,), "bfloat16")
            red = T.alloc_shared((THREADS,), "float32")
            sml = T.alloc_shared((4 * HQ + 4,), "float32")
            sel = T.alloc_shared((2 * KTOP,), "float32")
            bar = T.alloc_barrier([1] * 6)
            acc = T.alloc_local((1,), "float32")
            av = T.alloc_local((16,), "float32")
            fv = T.alloc_local((4,), "float32")
            iv = T.alloc_local((4,), "int32")
            # One accumulator per KV group: a gemm's output layout is built for
            # two dimensions and a slice of a three-dimensional fragment is not.
            of0 = T.alloc_fragment((GQA, DH), "float32")
            of1 = T.alloc_fragment((GQA, DH), "float32")
            sfg = T.alloc_fragment((GQA, ABLK), "float32")
            # Q is a gemm operand that never leaves the step, so it lives in
            # registers rather than in the shared budget: four per thread.
            qf0 = T.alloc_fragment((GQA, DH), "bfloat16")
            qf1 = T.alloc_fragment((GQA, DH), "bfloat16")
'''

TAIL = '''
    return mega


# ---------------------------------------------------------------------------
# Host side: one launch per step.
# ---------------------------------------------------------------------------

class Runner:
    """What the twin keeps between steps: the packed weights and the scratch."""

    def __init__(self, device):
        self.device = device
        self.packed = Packed(device)
        self.kernel = None
        self.scratch = torch.zeros(SCRATCH, dtype=torch.float32, device=device)
        self.logits = torch.zeros(V, dtype=torch.float32, device=device)
        self.lmeta = torch.tensor([x for m in _LAYER_META for x in m],
                                  dtype=torch.int32, device=device)
        # A ring, because nothing synchronises the host to the card between
        # steps any more: the table for the next step must not land in the
        # buffer the copy for this one is still reading. Four deep is as far
        # ahead as the host ever gets.
        self.slot = 0
        self.tab_host = [torch.zeros(NTAB, dtype=torch.int64).pin_memory()
                         for _ in range(RING)]
        self.tab_np = [t.numpy() for t in self.tab_host]
        self.tab = [torch.zeros(NTAB, dtype=torch.int64, device=device)
                    for _ in range(RING)]
        self.done = [torch.cuda.Event() for _ in range(RING)]
        for ev in self.done:
            ev.record()
        # Fresh Mamba state lands in the twin's own buffers, two deep so a step
        # never writes the tensors it was handed.
        self.pool = [[(torch.zeros(1, CONV, WIN, dtype=torch.bfloat16, device=device),
                       torch.zeros(1, MH, MD, SS, dtype=torch.float32, device=device))
                      for _ in range(N_MAMBA)] for _ in range(2)]
        self.parity = 0
        self.nlayer = NL

    def ensure(self):
        if self.kernel is None:
            self.kernel = build()
        return self.kernel


def run_step(run, args, cur_pos, index):
    """Fill the address table, launch once, hand back the logits and the state."""
    kernel = run.ensure()
    out = run.pool[run.parity]
    run.parity ^= 1
    sl = run.slot
    run.slot = (sl + 1) % RING
    run.done[sl].synchronize()
    tab = run.tab_np[sl]
    fresh = []
    at_m = at_a = 0
    ctx_full = ctx_tail = 0
    for layer, (kid, _) in enumerate(_LAYER_META):
        if kid == 0:
            tab[P_CONV_IN + at_m] = args[index[f"l{layer}_conv_state"]].data_ptr()
            tab[P_SSM_IN + at_m] = args[index[f"l{layer}_ssm_state"]].data_ptr()
            tab[P_CONV_OUT + at_m] = out[at_m][0].data_ptr()
            tab[P_SSM_OUT + at_m] = out[at_m][1].data_ptr()
            fresh += [out[at_m][0], out[at_m][1]]
            at_m += 1
        elif kid == 1:
            kc = args[index[f"l{layer}_k_cache"]]
            vc = args[index[f"l{layer}_v_cache"]]
            kt = args[index[f"l{layer}_k_tail"]]
            vt = args[index[f"l{layer}_v_tail"]]
            tab[P_KC + at_a] = kc.data_ptr()
            tab[P_VC + at_a] = vc.data_ptr()
            tab[P_KT + at_a] = kt.data_ptr()
            tab[P_VT + at_a] = vt.data_ptr()
            fresh += [kt[:, cur_pos:cur_pos + 1], vt[:, cur_pos:cur_pos + 1]]
            ctx_full, ctx_tail = kc.shape[1], kt.shape[1]
            at_a += 1
    tab[NPTR + 0] = ctx_full
    tab[NPTR + 1] = ctx_tail
    tab[NPTR + 2] = cur_pos
    tab[NPTR + 3] = run.nlayer
    tab[NPTR + 4] = ABLK
    run.tab[sl].copy_(run.tab_host[sl], non_blocking=True)
    kernel(*run.packed.flat_order(), run.tab[sl], run.lmeta,
           args[index["token_ids"]], run.scratch, run.logits)
    run.done[sl].record()
    return (run.logits.view(1, V), *fresh)
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mega_kernel.py")
    ap.add_argument("--prof", action="store_true",
                    help="build the twin that charges cycles to named stages")
    a = ap.parse_args()
    PROF = a.prof
    globals()["PROF"] = a.prof
    text = (HEADER.format(CTAS=CTAS, NHEAD=NHEAD, NW=NW, THREADS=THREADS, RB=RB, ARENA=ARENA,
                          HSPLIT=HSPLIT,
                          FROW=FROW, FARENA=FARENA, ABLK=ABLK)
            + body() + "\n" + TAIL
            + ATTN.format(head_body=attn_body("head"),
                          ctx_body=attn_body("context")))
    if a.prof:
        text = text.replace('F_BC = 0',
                            f'NPROBE = {len(PROBES)}\n'
                            f'PROBE_NAMES = {PROBES!r}\nF_BC = 0')
        text = text.replace('        logits: T.Tensor((V,), "float32"),\n    ):',
                            '        logits: T.Tensor((V,), "float32"),\n'
                            '        prof: T.Tensor((NPROBE + 1,), "int64"),\n    ):')
        text = text.replace("self.logits = torch.zeros",
                            "self.prof = torch.zeros(NPROBE + 1, dtype=torch.int64,"
                            " device=device)\n        self.logits = torch.zeros")
        text = text.replace("run.scratch, run.logits)", "run.scratch, run.logits, run.prof)")
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out} ({len(text.splitlines())} lines)"
          + (f", {len(PROBES)} probes" if a.prof else ""))
