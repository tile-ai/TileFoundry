"""TileLang kernels for one Qwen3-1.7B decode step.

Every kernel takes caller-owned output buffers -- no allocation, and no shape or
trip count that depends on a host value -- because the decode loop replays them
from a captured CUDA graph. The step's position is read from a one-element
device tensor (``Pos``) instead of being baked in, which is what lets a single
capture serve every step of a generation.

Two shapes recur and set the tiling:

* **GEMV.** Decode is one token, so every projection is ``(K,) @ (K, N)`` --
  pure memory traffic, no reuse. A block owns ``BN`` output columns and walks
  ``K``; weights are stored ``(K, N)``, so a row slice is contiguous and
  coalesces. ``N/BN`` blocks alone leave most of an H200's 132 SMs idle on the
  small projections, so ``K`` is split ``SK`` ways as well and the partial sums
  land in an ``(SK, N)`` f32 buffer that the *consumer* reduces -- the reduce is
  never a kernel of its own.

* **Two-pass attention.** One query row against ``pos+1`` cached keys. A block
  owns ``SS`` context positions: pass one reads K and fills the whole score row,
  then the row is normalised, then pass two reads V and weights it. Holding the
  split's entire score row means no rescale happens inside a block -- only
  across blocks, which ``gemv_attn_combine`` folds into ``o_proj``. A block whose
  slice starts past the current length writes a neutral partial and exits, so a
  fixed grid still costs only the context that exists.

Two TileLang facts shape the code below and are worth stating once:

* A fragment lives in the owning thread's registers. Anything one thread must
  read that another wrote goes through ``alloc_shared``; ``T.reduce_*`` is the
  exception, because its one-element result is replicated and so readable by
  every thread.
* A kernel body is source-rewritten, and a plain helper function is not. So
  loop-emitting code is written inline here rather than factored out -- a
  ``T.Parallel`` inside an ordinary callee would execute as Python and fail.
"""
from functools import lru_cache

import tilelang
import tilelang.language as T

DT = "bfloat16"
ACC = "float32"
NEG = -1.0e30

tilelang.disable_cache()


@lru_cache(maxsize=None)
def _compile(prim):
    return tilelang.compile(prim)


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def rms_norm(H: int, eps: float, threads: int = 256):
    """``Xn = bf16(x * rsqrt(mean(x^2) + eps)) * gamma``, one block.

    Qwen3RMSNorm rounds to the input dtype *before* the learned scale, so the
    cast sits inside the product rather than after it -- the order the authored
    HIR spells out, and not the one a generic rms_norm would take.
    """
    @T.prim_func
    def main(X: T.Tensor((H,), DT), G: T.Tensor((H,), DT), Xn: T.Tensor((H,), DT)):
        with T.Kernel(1, threads=threads):
            xs = T.alloc_fragment((H,), ACC)
            sq = T.alloc_fragment((H,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            for i in T.Parallel(H):
                xs[i] = X[i].astype(ACC)
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, tot, dim=0)
            for i in T.Parallel(H):
                Xn[i] = (xs[i] * T.rsqrt(tot[0] / H + eps)).astype(DT) * G[i]

    return _compile(main)


@lru_cache(maxsize=None)
def resid_rms_norm(H: int, SK: int, eps: float, threads: int = 256):
    """``Hout = A + reduce(P)``; ``Xn = norm(Hout) * gamma``. One block.

    The residual add and the norm that follows it are the same read of the same
    vector, so they are one kernel -- and *P*, the producer's split-K partial,
    is reduced here rather than by a kernel of its own.
    """
    @T.prim_func
    def main(
        A: T.Tensor((H,), DT),
        P: T.Tensor((SK, H), ACC),
        G: T.Tensor((H,), DT),
        Hout: T.Tensor((H,), DT),
        Xn: T.Tensor((H,), DT),
    ):
        with T.Kernel(1, threads=threads):
            xs = T.alloc_fragment((H,), ACC)
            sq = T.alloc_fragment((H,), ACC)
            acc = T.alloc_fragment((H,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            T.clear(acc)
            for s in T.serial(SK):
                for i in T.Parallel(H):
                    acc[i] += P[s, i]
            for i in T.Parallel(H):
                v = A[i] + acc[i].astype(DT)
                Hout[i] = v
                xs[i] = v.astype(ACC)
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, tot, dim=0)
            for i in T.Parallel(H):
                Xn[i] = (xs[i] * T.rsqrt(tot[0] / H + eps)).astype(DT) * G[i]

    return _compile(main)


# --------------------------------------------------------------------------- #
# GEMV
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def gemv(K: int, N: int, BN: int, BK: int, SK: int, threads: int, stages: int = 3):
    """``P[s, n] = sum(X[k] * W[k, n])`` over split *s*'s share of ``K``."""
    KS = K // SK

    @T.prim_func
    def main(X: T.Tensor((K,), DT), W: T.Tensor((K, N), DT), P: T.Tensor((SK, N), ACC)):
        with T.Kernel(T.ceildiv(N, BN), SK, threads=threads) as (bx, bs):
            Ws = T.alloc_shared((BK, BN), DT)
            Xs = T.alloc_shared((BK,), DT)
            acc = T.alloc_fragment((BN,), ACC)
            T.clear(acc)
            for ko in T.Pipelined(KS // BK, num_stages=stages):
                k0 = bs * KS + ko * BK
                T.copy(W[k0:k0 + BK, bx * BN:(bx + 1) * BN], Ws)
                T.copy(X[k0:k0 + BK], Xs)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += Xs[kk].astype(ACC) * Ws[kk, j].astype(ACC)
            for j in T.Parallel(BN):
                P[bs, bx * BN + j] = acc[j]

    return _compile(main)


#: Two GEMVs below fold their producer in rather than reading a materialised
#: vector. The reason is not the traffic saved -- these vectors are a few KB --
#: but the kernel saved. A split-K block only ever consumes ``K / SK`` of its
#: input, and for both of these that slice can be built from partials in L2, so
#: the producer stops being a separate launch with its own ramp and drain.

@lru_cache(maxsize=None)
def gemv_attn_combine(
    HQ: int, D: int, N: int, BN: int, BK: int, SK: int, NS: int,
    threads: int, stages: int = 3,
):
    """``o_proj``, with the attention splits merged into its own input read.

    A block owning ``K / SK`` inputs owns a whole number of attention heads
    (``K = HQ * D`` and the split divides the head count), so it can merge just
    those heads' partials itself. Each thread redoes the ``NS``-term log-sum-exp
    for its own entry -- the partials are a few KB and sit in L2.
    """
    K = HQ * D
    KS = K // SK

    @T.prim_func
    def main(
        Op: T.Tensor((NS, HQ, D), ACC),
        Mp: T.Tensor((NS, HQ), ACC),
        Lp: T.Tensor((NS, HQ), ACC),
        W: T.Tensor((K, N), DT),
        P: T.Tensor((SK, N), ACC),
    ):
        with T.Kernel(T.ceildiv(N, BN), SK, threads=threads) as (bx, bs):
            Ws = T.alloc_shared((BK, BN), DT)
            Xs = T.alloc_shared((KS,), DT)
            mx = T.alloc_fragment((KS,), ACC)
            den = T.alloc_fragment((KS,), ACC)
            num = T.alloc_fragment((KS,), ACC)
            acc = T.alloc_fragment((BN,), ACC)
            for i in T.Parallel(KS):
                h = (bs * KS + i) // D
                d = (bs * KS + i) % D
                mx[i] = NEG
                for s in T.serial(NS):
                    mx[i] = T.max(mx[i], Mp[s, h])
                den[i] = 0.0
                num[i] = 0.0
                for s in T.serial(NS):
                    den[i] += Lp[s, h] * T.exp(Mp[s, h] - mx[i])
                    num[i] += Op[s, h, d] * T.exp(Mp[s, h] - mx[i])
                Xs[i] = (num[i] / den[i]).astype(DT)
            T.sync_threads()
            T.clear(acc)
            for ko in T.Pipelined(KS // BK, num_stages=stages):
                k0 = bs * KS + ko * BK
                T.copy(W[k0:k0 + BK, bx * BN:(bx + 1) * BN], Ws)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += Xs[ko * BK + kk].astype(ACC) * Ws[kk, j].astype(ACC)
            for j in T.Parallel(BN):
                P[bs, bx * BN + j] = acc[j]

    return _compile(main)


@lru_cache(maxsize=None)
def gemv_silu(
    I: int, N: int, BN: int, BK: int, SK: int, SKG: int, threads: int, stages: int = 3
):
    """``down_proj``, with ``silu(gate) * up`` folded into its own input read.

    *GU* is the fused gate/up GEMV's partial, gate in ``[0, I)`` and up in
    ``[I, 2I)``. A block reduces and activates only the ``I / SK`` entries it
    walks, so the intermediate never reaches HBM in either direction.
    """
    KS = I // SK

    @T.prim_func
    def main(
        GU: T.Tensor((SKG, 2 * I), ACC),
        W: T.Tensor((I, N), DT),
        P: T.Tensor((SK, N), ACC),
    ):
        with T.Kernel(T.ceildiv(N, BN), SK, threads=threads) as (bx, bs):
            Ws = T.alloc_shared((BK, BN), DT)
            Xs = T.alloc_shared((KS,), DT)
            g = T.alloc_fragment((KS,), ACC)
            u = T.alloc_fragment((KS,), ACC)
            acc = T.alloc_fragment((BN,), ACC)
            T.clear(g)
            T.clear(u)
            for s in T.serial(SKG):
                for i in T.Parallel(KS):
                    g[i] += GU[s, bs * KS + i]
                    u[i] += GU[s, I + bs * KS + i]
            for i in T.Parallel(KS):
                gb = g[i].astype(DT).astype(ACC)
                ub = u[i].astype(DT).astype(ACC)
                Xs[i] = (
                    (gb / (1.0 + T.exp(-gb))).astype(DT).astype(ACC) * ub
                ).astype(DT)
            T.sync_threads()
            T.clear(acc)
            for ko in T.Pipelined(KS // BK, num_stages=stages):
                k0 = bs * KS + ko * BK
                T.copy(W[k0:k0 + BK, bx * BN:(bx + 1) * BN], Ws)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += Xs[ko * BK + kk].astype(ACC) * Ws[kk, j].astype(ACC)
            for j in T.Parallel(BN):
                P[bs, bx * BN + j] = acc[j]

    return _compile(main)


@lru_cache(maxsize=None)
def lm_head(K: int, N: int, BN: int, BK: int, threads: int, stages: int = 3):
    """The head, plus each block's own best entry.

    Greedy sampling wants one index out of 152k, and a block already holds its
    columns in registers -- so it reduces them here and writes ``(value,
    index)``, leaving a 1187-entry reduction instead of a second full pass over
    the logits. The logits are written too: they cost one store and are what a
    comparison against the reference reads.
    """
    NB = (N + BN - 1) // BN

    @T.prim_func
    def main(
        X: T.Tensor((K,), DT),
        W: T.Tensor((K, N), DT),
        O: T.Tensor((N,), ACC),
        Bv: T.Tensor((NB,), ACC),
        Bi: T.Tensor((NB,), "int32"),
    ):
        with T.Kernel(NB, threads=threads) as bx:
            Ws = T.alloc_shared((BK, BN), DT)
            Xs = T.alloc_shared((BK,), DT)
            acc = T.alloc_fragment((BN,), ACC)
            stg = T.alloc_shared((BN,), ACC)
            red = T.alloc_fragment((BN,), ACC)
            sel = T.alloc_fragment((BN,), ACC)
            mx = T.alloc_fragment((1,), ACC)
            win = T.alloc_fragment((1,), ACC)
            T.clear(acc)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(W[ko * BK:(ko + 1) * BK, bx * BN:(bx + 1) * BN], Ws)
                T.copy(X[ko * BK:(ko + 1) * BK], Xs)
                for j in T.Parallel(BN):
                    for kk in T.serial(BK):
                        acc[j] += Xs[kk].astype(ACC) * Ws[kk, j].astype(ACC)
            # The block's own best, found without a second pass over the logits.
            # Two things matter about how it is written:
            #
            # * `acc` is staged to shared and the reductions read a *fresh*
            #   fragment. Reducing `acc` twice in place makes layout inference
            #   replicate it across all 128 threads to satisfy both uses, which
            #   spills the accumulator and costs 19ms instead of 165us.
            # * Argmax is two max-reductions, no atomic: the winning value, then
            #   `BN - j` over the entries attaining it, whose max is the *lowest*
            #   winning index -- torch.argmax's tie-break. An `atomic_min` on
            #   shared costs 18ms here; a reduce_max costs 2us.
            T.copy(acc, stg)
            for j in T.Parallel(BN):
                O[bx * BN + j] = stg[j]
            T.sync_threads()
            for j in T.Parallel(BN):
                red[j] = stg[j]
            T.reduce_max(red, mx, dim=0)
            for j in T.Parallel(BN):
                sel[j] = T.if_then_else(red[j] >= mx[0], (BN - j) * 1.0, 0.0)
            T.reduce_max(sel, win, dim=0)
            if T.get_thread_binding() == 0:
                Bv[bx] = mx[0]
                Bi[bx] = bx * BN + BN - T.Cast("int32", win[0])

    return _compile(main), NB


# --------------------------------------------------------------------------- #
# q/k norm + rope + cache write
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def qk_rope_cache(HQ: int, HKV: int, D: int, MP: int, CAP: int, SK: int, eps: float):
    """Reduce the fused QKV partials, then per-head norm, rope, and cache write.

    *P* is laid out ``[q | k | v]`` over its last axis, one block per head of
    the three. Query heads keep going, into *Q*. Key and value heads stop here:
    this step's entry is written straight into the cache at ``Pos``, so no
    caller appends anything and the cache buffer never moves, which is what
    makes the step replayable from a fixed graph.

    ``Pos`` is the rotary position and ``Wr`` the slot to write. Decoding passes
    the same tensor twice, but the authored reference lets them differ -- it
    takes ``pos_ids`` and a prior cache of its own length -- so they are separate
    parameters, and the twin that stands for that reference can honour it.

    The ``1/sqrt(head_dim)`` factor is deliberately *not* applied here. The
    authored reference multiplies it onto ``q`` in bf16 before the dot, which
    rounds every entry once more -- up to 2^-9 relative, and the exponential
    downstream turns that into percent-level error on the attention weights.
    Hugging Face scales after the dot instead, so ``attn_partial`` folds the
    factor into its f32 accumulation.
    """
    QN, KN = HQ * D, HKV * D
    TOT = HQ + 2 * HKV

    @T.prim_func
    def main(
        P: T.Tensor((SK, QN + 2 * KN), ACC),
        Gq: T.Tensor((D,), DT),
        Gk: T.Tensor((D,), DT),
        Cos: T.Tensor((MP, D), DT),
        Sin: T.Tensor((MP, D), DT),
        Pos: T.Tensor((1,), "int32"),
        Wr: T.Tensor((1,), "int32"),
        Kc: T.Tensor((CAP, KN), DT),
        Vc: T.Tensor((CAP, KN), DT),
        Q: T.Tensor((HQ * D,), DT),
    ):
        with T.Kernel(TOT, threads=D) as bh:
            xs = T.alloc_fragment((D,), ACC)
            sq = T.alloc_fragment((D,), ACC)
            acc = T.alloc_fragment((D,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            nrm = T.alloc_shared((D,), DT)      # rope pairs d with d +/- D/2
            head = T.if_then_else(
                bh < HQ, bh, T.if_then_else(bh < HQ + HKV, bh - HQ, bh - HQ - HKV)
            )
            base = T.if_then_else(
                bh < HQ, 0, T.if_then_else(bh < HQ + HKV, QN, QN + KN)
            ) + head * D
            T.clear(acc)
            for s in T.serial(SK):
                for d in T.Parallel(D):
                    acc[d] += P[s, base + d]
            for d in T.Parallel(D):
                xs[d] = acc[d].astype(DT).astype(ACC)
                sq[d] = xs[d] * xs[d]
            with T.If(bh >= HQ + HKV):
                with T.Then():
                    for d in T.Parallel(D):
                        Vc[Wr[0], head * D + d] = xs[d].astype(DT)
                with T.Else():
                    T.reduce_sum(sq, tot, dim=0)
                    for d in T.Parallel(D):
                        g = T.if_then_else(bh < HQ, Gq[d], Gk[d])
                        nrm[d] = (xs[d] * T.rsqrt(tot[0] / D + eps)).astype(DT) * g
                    T.sync_threads()
                    for d in T.Parallel(D):
                        half = T.if_then_else(
                            d < D // 2,
                            -nrm[d + D // 2].astype(ACC),
                            nrm[d - D // 2].astype(ACC),
                        )
                        rot = (
                            nrm[d].astype(ACC) * Cos[Pos[0], d].astype(ACC)
                            + half * Sin[Pos[0], d].astype(ACC)
                        ).astype(DT)
                        with T.If(bh < HQ):
                            with T.Then():
                                Q[head * D + d] = rot
                            with T.Else():
                                Kc[Wr[0], head * D + d] = rot

    return _compile(main)


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def attn_partial(
    HQ: int, HKV: int, D: int, CAP: int, SS: int, scale: float, threads: int = 128
):
    """One split's ``(max, sum, weighted values)`` over its slice of the context.

    Grid is ``(splits, kv heads)``: one kv head serves its whole GQA group, so a
    key is read once and used by every query head that shares it, rather than
    being materialised per head the way a literal reading of the reference's
    ``repeat_interleave`` would.

    *scale* is applied to the finished f32 dot product, not to ``q`` beforehand
    -- see ``qk_rope_cache`` for why that ordering matters here.

    Both products go through ``T.gemm``, which means padding the group's ``G``
    query rows up to MMA's smallest ``M`` of 16 and leaving the rest zero. The
    8x arithmetic waste costs nothing -- one query row against a few hundred
    keys is latency-bound, not flop-bound -- and it buys TileLang's own MMA
    shared-memory layouts. Written as a scalar loop instead, consecutive threads
    read one row apart, every address lands in the same bank, and the kernel
    runs at 0.4 TB/s; this way it is 2.2x quicker.
    """
    G = HQ // HKV
    NS = (CAP + SS - 1) // SS
    M = 16

    @T.prim_func
    def main(
        Q: T.Tensor((HQ * D,), DT),
        Kc: T.Tensor((CAP, HKV * D), DT),
        Vc: T.Tensor((CAP, HKV * D), DT),
        Pos: T.Tensor((1,), "int32"),
        Op: T.Tensor((NS, HQ, D), ACC),
        Mp: T.Tensor((NS, HQ), ACC),
        Lp: T.Tensor((NS, HQ), ACC),
    ):
        with T.Kernel(NS, HKV, threads=threads) as (bs, bh):
            Qs = T.alloc_shared((M, D), DT)
            Ks = T.alloc_shared((SS, D), DT)
            Vs = T.alloc_shared((SS, D), DT)
            sc = T.alloc_fragment((M, SS), ACC)
            scb = T.alloc_shared((M, SS), DT)
            orun = T.alloc_fragment((M, D), ACC)
            mx = T.alloc_fragment((M,), ACC)
            sm = T.alloc_fragment((M,), ACC)
            with T.If(bs * SS < Pos[0] + 1):
                with T.Then():
                    for m, d in T.Parallel(M, D):
                        Qs[m, d] = T.if_then_else(
                            m < G,
                            Q[(bh * G + T.min(m, G - 1)) * D + d],
                            T.Cast(DT, 0.0),
                        )
                    T.copy(Kc[bs * SS:(bs + 1) * SS, bh * D:(bh + 1) * D], Ks)
                    T.copy(Vc[bs * SS:(bs + 1) * SS, bh * D:(bh + 1) * D], Vs)
                    T.clear(sc)
                    T.gemm(Qs, Ks, sc, transpose_B=True)
                    for m, s in T.Parallel(M, SS):
                        # past the end of the context this position does not exist
                        sc[m, s] = T.if_then_else(
                            bs * SS + s < Pos[0] + 1, sc[m, s] * scale, NEG
                        )
                    T.reduce_max(sc, mx, dim=1, clear=True)
                    for m, s in T.Parallel(M, SS):
                        sc[m, s] = T.exp(sc[m, s] - mx[m])
                    T.reduce_sum(sc, sm, dim=1, clear=True)
                    # bf16 probabilities into the second product, which is what
                    # HF's own attention does before it multiplies by V
                    T.copy(sc, scb)
                    T.clear(orun)
                    T.gemm(scb, Vs, orun)
                    for g, d in T.Parallel(G, D):
                        Op[bs, bh * G + g, d] = orun[g, d]
                    for g in T.Parallel(G):
                        Mp[bs, bh * G + g] = mx[g]
                        Lp[bs, bh * G + g] = sm[g]
                with T.Else():
                    for g, d in T.Parallel(G, D):
                        Op[bs, bh * G + g, d] = 0.0
                    for g in T.Parallel(G):
                        Mp[bs, bh * G + g] = NEG
                        Lp[bs, bh * G + g] = 0.0

    return _compile(main)


@lru_cache(maxsize=None)
def attn_combine(HQ: int, D: int, CAP: int, SS: int):
    """Merge the splits' partials against their joint max; head-major flatten.

    Head-major is not a choice: ``w_o`` was stored expecting attention entry
    ``(h, d)`` at ``h * D + d``, matching the authored reshape.
    """
    NS = (CAP + SS - 1) // SS

    @T.prim_func
    def main(
        Op: T.Tensor((NS, HQ, D), ACC),
        Mp: T.Tensor((NS, HQ), ACC),
        Lp: T.Tensor((NS, HQ), ACC),
        O: T.Tensor((HQ * D,), DT),
    ):
        with T.Kernel(HQ, threads=D) as bh:
            mx = T.alloc_fragment((D,), ACC)
            den = T.alloc_fragment((D,), ACC)
            acc = T.alloc_fragment((D,), ACC)
            # Every thread needs the joint max and denominator, and there are
            # only NS of each -- so each recomputes them from L2 rather than one
            # reducing and broadcasting. A cross-thread reduce over NS values
            # has no layout anyway when NS does not divide the thread count.
            for d in T.Parallel(D):
                mx[d] = NEG
                for s in T.serial(NS):
                    mx[d] = T.max(mx[d], Mp[s, bh])
                den[d] = 0.0
                acc[d] = 0.0
                for s in T.serial(NS):
                    e = T.exp(Mp[s, bh] - mx[d])
                    den[d] += Lp[s, bh] * e
                    acc[d] += Op[s, bh, d] * e
                O[bh * D + d] = (acc[d] / den[d]).astype(DT)

    return _compile(main)


# --------------------------------------------------------------------------- #
# MLP activation
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def silu_mul(I: int, SK: int, BN: int = 256, threads: int = 256):
    """Reduce the fused gate/up partials, then ``silu(gate) * up``.

    *P* holds gate in ``[0, I)`` and up in ``[I, 2I)``: one GEMV produced both,
    so one kernel consumes both and neither reaches HBM on its own.
    """
    @T.prim_func
    def main(P: T.Tensor((SK, 2 * I), ACC), O: T.Tensor((I,), DT)):
        with T.Kernel(T.ceildiv(I, BN), threads=threads) as bx:
            g = T.alloc_fragment((BN,), ACC)
            u = T.alloc_fragment((BN,), ACC)
            T.clear(g)
            T.clear(u)
            for s in T.serial(SK):
                for j in T.Parallel(BN):
                    g[j] += P[s, bx * BN + j]
                    u[j] += P[s, I + bx * BN + j]
            for j in T.Parallel(BN):
                gb = g[j].astype(DT).astype(ACC)
                ub = u[j].astype(DT).astype(ACC)
                O[bx * BN + j] = (
                    (gb / (1.0 + T.exp(-gb))).astype(DT).astype(ACC) * ub
                ).astype(DT)

    return _compile(main)


# --------------------------------------------------------------------------- #
# embedding and sampling
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def embed(V: int, H: int, threads: int = 256):
    """The decoded token's own row of the table."""
    @T.prim_func
    def main(Tbl: T.Tensor((V, H), DT), Ids: T.Tensor((1,), "int64"), O: T.Tensor((H,), DT)):
        with T.Kernel(T.ceildiv(H, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                O[bx * threads + i] = Tbl[Ids[0], bx * threads + i]

    return _compile(main)


@lru_cache(maxsize=None)
def sample_step(NB: int, NSTEPS: int, threads: int = 256):
    """Finish the greedy pick, record it, hand the next input on, advance ``Pos``.

    This is the graph's last node and the only one that decides anything: while
    the prompt still has a token left it feeds that, otherwise it feeds what it
    just sampled. That one device-side choice is what lets a single capture
    serve the prompt walk and the continuation with no host round trip between
    steps.
    """
    PAD = ((NB + threads - 1) // threads) * threads

    @T.prim_func
    def main(
        Bv: T.Tensor((NB,), ACC),
        Bi: T.Tensor((NB,), "int32"),
        Inp: T.Tensor((NSTEPS,), "int32"),
        PromptLen: T.Tensor((1,), "int32"),
        Ids: T.Tensor((1,), "int64"),
        Pos: T.Tensor((1,), "int32"),
        Sam: T.Tensor((NSTEPS,), "int32"),
    ):
        with T.Kernel(1, threads=threads):
            # Padded to a whole number of thread-rounds: a fragment whose size
            # does not divide the thread count has no reduce layout.
            f = T.alloc_fragment((PAD,), ACC)
            sel = T.alloc_fragment((PAD,), ACC)
            mx = T.alloc_fragment((1,), ACC)
            win = T.alloc_fragment((1,), ACC)
            for k in T.Parallel(PAD):
                f[k] = T.if_then_else(k < NB, Bv[T.min(k, NB - 1)], NEG)
            T.reduce_max(f, mx, dim=0)
            for k in T.Parallel(PAD):
                sel[k] = T.if_then_else(
                    (k < NB) and (f[k] >= mx[0]), (NB - k) * 1.0, 0.0
                )
            T.reduce_max(sel, win, dim=0)
            if T.get_thread_binding() == 0:
                p = Pos[0]
                best = Bi[NB - T.Cast("int32", win[0])]
                Sam[p] = best
                Ids[0] = T.if_then_else(
                    p + 1 < PromptLen[0], Inp[p + 1].astype("int64"), best.astype("int64")
                )
                Pos[0] = p + 1

    return _compile(main)
