"""RoPE, assembly and MLA decode attention -- one kernel, one block per head.

Everything between the two up-projections and `o_proj` is per-head and tiny, so
splitting it into a rotate kernel, a concat kernel and an attention kernel would
be three boundaries for work that never leaves a head:

    q_b's 96 numbers for head h -> rotate the last 32 -> attend -> 64 numbers

so it is one kernel. Block *h* rotates its own query, assembles its own key from
`kv_b`'s nope half and the one rotary slice every head shares, and attends the
cache together with the token it just built.

── The two things that are not obvious ──────────────────────────────────────

**The context length is read from device memory, not passed as a number.** A
captured CUDA graph freezes its kernels' arguments, so a `ctx` passed as a
scalar would be whatever it was at capture and every later step would attend the
wrong prefix. `ctx` is an `int32[1]` tensor the loop reads, and the caches
arrive as flat, layout-dynamic tensors so their extent is not baked either.

**The softmax is online and merged across warps, not taken over a
concatenation.** The cached positions and this token's own live in differently
shaped places -- `ctx` rows of a cache versus 96 numbers just computed -- and
the reference splits them for the same reason. Each warp walks a stride of the
context keeping its own `(max, sum, weighted)`, warp 0 folds in the new token
from shared memory, and a log-sum-exp rescale merges the eight partials.

**Nothing here writes the cache.** The step hands its key and value back and
appending them is the caller's, which is the authored Module's contract and
also what keeps this kernel safe to point at any cache `check` invents.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cuda.bindings import driver as _cuda

from ._common import BF16, F32, Compiled

#: Eight warps per head: eight cached positions scored at once, which is what
#: keeps a 40-block grid from being pure latency on a short context.
AWARPS = 8
ANT = AWARPS * 32

#: Stands in for -inf as a softmax running maximum. A real -inf would make the
#: first rescale `exp(-inf - -inf)` = NaN in the warps that see no positions.
NEG_INF = -3.0e38

#: log2(e). The softmax is written with `exp2(x * LOG2E)` because that is one
#: `ex2.approx.ftz.f32` instruction, where `exp` is a call; a decode step runs
#: two of them per cached position per head per layer.
LOG2E = 1.4426950408889634


def _make_rope_attend(heads: int, qk: int, nope: int, rope: int, v_dim: int):
    half = rope // 2
    kv_pair = nope + v_dim

    @cute.kernel
    def kern(
        attn: cute.Tensor, k_new: cute.Tensor, v_new: cute.Tensor,
        q_up: cute.Tensor, kv_up: cute.Tensor, k_rope: cute.Tensor,
        cos: cute.Tensor, sin: cute.Tensor, pos: cute.Tensor,
        k_cache: cute.Tensor, v_cache: cute.Tensor, ctx: cute.Tensor,
        scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        h, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32

        sq = cute.make_tensor(cute.arch.alloc_smem(F32, qk, 16), cute.make_layout(qk))
        sk = cute.make_tensor(cute.arch.alloc_smem(F32, qk, 16), cute.make_layout(qk))
        sv = cute.make_tensor(cute.arch.alloc_smem(F32, v_dim, 16),
                              cute.make_layout(v_dim))
        sm = cute.make_tensor(cute.arch.alloc_smem(F32, AWARPS, 16),
                              cute.make_layout(AWARPS))
        sl = cute.make_tensor(cute.arch.alloc_smem(F32, AWARPS, 16),
                              cute.make_layout(AWARPS))
        sa = cute.make_tensor(cute.arch.alloc_smem(F32, AWARPS * v_dim, 16),
                              cute.make_layout((AWARPS, v_dim), stride=(v_dim, 1)))

        p = pos[0]
        n = ctx[0]
        sc = scale[0].to(F32)

        # ── this head's query and key ────────────────────────────────────────
        # Three disjoint ranges of `tidx`, each writing only its own slots, so
        # no value has to survive a dynamic branch.
        if tidx < nope:
            # The nope halves pass straight through, `q * scale` apart.
            q_bf = q_up[h * qk + tidx]
            k_bf = kv_up[h * kv_pair + tidx]
            sq[tidx] = (q_bf.to(F32) * sc).to(BF16).to(F32)
            sk[tidx] = k_bf.to(F32)
            k_new[h * qk + tidx] = k_bf
        elif tidx < nope + half:
            # rotate_half's lower half: paired with r + half, which is negated.
            r = tidx - nope
            c = cos[p, r].to(F32)
            s = sin[p, r].to(F32)
            q_bf = (q_up[h * qk + nope + r].to(F32) * c
                    - q_up[h * qk + nope + r + half].to(F32) * s).to(BF16)
            k_bf = (k_rope[r].to(F32) * c - k_rope[r + half].to(F32) * s).to(BF16)
            sq[nope + r] = (q_bf.to(F32) * sc).to(BF16).to(F32)
            sk[nope + r] = k_bf.to(F32)
            k_new[h * qk + nope + r] = k_bf
        elif tidx < qk:
            # rotate_half's upper half: paired with r - half, taken positive.
            r = tidx - nope
            c = cos[p, r].to(F32)
            s = sin[p, r].to(F32)
            q_bf = (q_up[h * qk + nope + r].to(F32) * c
                    + q_up[h * qk + nope + r - half].to(F32) * s).to(BF16)
            k_bf = (k_rope[r].to(F32) * c + k_rope[r - half].to(F32) * s).to(BF16)
            sq[nope + r] = (q_bf.to(F32) * sc).to(BF16).to(F32)
            sk[nope + r] = k_bf.to(F32)
            k_new[h * qk + nope + r] = k_bf
        if tidx < v_dim:
            val = kv_up[h * kv_pair + nope + tidx]
            sv[tidx] = val.to(F32)
            v_new[h * v_dim + tidx] = val
        cute.arch.barrier()

        # ── online softmax over the cache, one stride per warp ───────────────
        m = F32(NEG_INF)
        lsum = F32(0.0)
        a0 = F32(0.0)
        a1 = F32(0.0)
        # Unrolled by four: without it each iteration is one memory round trip
        # that nothing overlaps -- the loads for the next position do not depend
        # on this one's score, but a rolled loop cannot start them early.
        for tpos in cutlass.range(warp, n, AWARPS, unroll=4):
            base = (tpos * heads + h) * qk
            part = F32(0.0)
            for j in cutlass.range_constexpr(qk // 32):
                d = lane + 32 * j
                part += sq[d] * k_cache[base + d].to(F32)
            score = cute.arch.warp_reduction_sum(part)
            m2 = cute.arch.fmax(m, score)
            corr = F32(cmath.exp2((m - m2) * LOG2E, fastmath=True))
            pw = F32(cmath.exp2((score - m2) * LOG2E, fastmath=True))
            vbase = (tpos * heads + h) * v_dim
            lsum = lsum * corr + pw
            a0 = a0 * corr + pw * v_cache[vbase + lane].to(F32)
            a1 = a1 * corr + pw * v_cache[vbase + lane + 32].to(F32)
            m = m2

        # This token attends itself. It is not in the cache -- appending is the
        # caller's step -- so warp 0 folds it in from what the block just built.
        if warp == 0:
            part = F32(0.0)
            for j in cutlass.range_constexpr(qk // 32):
                d = lane + 32 * j
                part += sq[d] * sk[d]
            score = cute.arch.warp_reduction_sum(part)
            m2 = cute.arch.fmax(m, score)
            corr = F32(cmath.exp2((m - m2) * LOG2E, fastmath=True))
            pw = F32(cmath.exp2((score - m2) * LOG2E, fastmath=True))
            lsum = lsum * corr + pw
            a0 = a0 * corr + pw * sv[lane]
            a1 = a1 * corr + pw * sv[lane + 32]
            m = m2

        # ── merge the warps' partials against their joint maximum ────────────
        if lane == 0:
            sm[warp] = m
            sl[warp] = lsum
        sa[warp, lane] = a0
        sa[warp, lane + 32] = a1
        cute.arch.barrier()

        if tidx < v_dim:
            peak = F32(NEG_INF)
            for w in cutlass.range_constexpr(AWARPS):
                peak = cute.arch.fmax(peak, sm[w])
            total = F32(0.0)
            weighted = F32(0.0)
            for w in cutlass.range_constexpr(AWARPS):
                rescale = F32(cmath.exp2((sm[w] - peak) * LOG2E, fastmath=True))
                total += sl[w] * rescale
                weighted += sa[w, tidx] * rescale
            attn[h * v_dim + tidx] = (weighted / total).to(BF16)

    @cute.jit
    def entry(attn: cute.Tensor, k_new: cute.Tensor, v_new: cute.Tensor,
              q_up: cute.Tensor, kv_up: cute.Tensor, k_rope: cute.Tensor,
              cos: cute.Tensor, sin: cute.Tensor, pos: cute.Tensor,
              k_cache: cute.Tensor, v_cache: cute.Tensor, ctx: cute.Tensor,
              scale: cute.Tensor, stream: _cuda.CUstream):
        kern(attn, k_new, v_new, q_up, kv_up, k_rope, cos, sin, pos,
             k_cache, v_cache, ctx, scale).launch(
            grid=[heads, 1, 1], block=[ANT, 1, 1], stream=stream)

    return entry


#: The two caches (positions 9 and 10) carry the only extent that changes.
ROPE_ATTEND = Compiled(_make_rope_attend, dynamic=(9, 10))

__all__ = ["ROPE_ATTEND"]
