"""The small kernels around the GEMMs: RMSNorm, embedding gather, the lm_head GEMV.

Rounding contracts (model.py):
  rmsnorm  y = bf16(f32(x) * rsqrt(mean(f32(x)^2) + eps) * (1 + f32(gamma)))   (HF Qwen3_5RMSNorm)
  embed    y[t] = table[ids[t]]
  gemv     y[n] = bf16(sum_k f32(x[k]) * f32(w[n, k]))                        (lm_head, one row)
"""
from functools import lru_cache

import tilelang
import tilelang.language as T

EPS = 1e-6


def _compile(func, name=None):
    if name:
        func = func.with_attr("global_symbol", name)
    return tilelang.compile(func, target="cuda")


@lru_cache(maxsize=None)
def rmsnorm_kernel(M: int, H: int, rows: int = 4):
    """One warp per row, ``rows`` rows per CTA. The f32 sum of squares runs in exactly
    torch's reduction order for ``x.float().pow(2).mean(-1)`` on a contiguous (M, H)
    tensor (the HIR evaluator and transformers both use it), so the output is bitwise
    equal to the HIR's tf.rms_norm: lane l owns the 4-element chunks l, l + 32, ...,
    keeps one accumulator per chunk position, adds them ((a0 + a1) + a2) + a3, then
    shfl_down 16, 8, 4, 2, 1. Found by emulation on 5504 real rows (0 mismatches; the
    shfl order 1, 2, 4, 8, 16 mismatches 39 % of rows). Every f32 op is IEEE (no FMA)."""
    dt, F32 = "bfloat16", "float32"
    rows = min(rows, M)
    nch = H // 128                    # 4-element chunks per lane
    assert H % 128 == 0 and M % rows == 0

    @T.prim_func
    def main(X: T.Tensor((M, H), dt), G: T.Tensor((H,), dt), Y: T.Tensor((M, H), dt)):
        with T.Kernel(M // rows, threads=32 * rows) as bid:
            tx = T.get_thread_binding()
            lane = tx % 32
            row = bid * rows + tx // 32
            xl = T.alloc_local((nch * 4,), dt)
            gl = T.alloc_local((nch * 4,), dt)
            acc = T.alloc_local((4,), F32)
            for c in T.serial(nch):
                for i in T.vectorized(4):
                    xl[c * 4 + i] = X[row, (lane + 32 * c) * 4 + i]
            for c in T.serial(nch):
                for i in T.vectorized(4):
                    gl[c * 4 + i] = G[(lane + 32 * c) * 4 + i]
            for i in T.unroll(4):
                acc[i] = T.float32(0)
            for c in T.serial(nch):
                for i in T.unroll(4):
                    xf = T.cast(xl[c * 4 + i], F32)
                    acc[i] = T.ieee_add(acc[i], T.ieee_mul(xf, xf))
            s = T.ieee_add(T.ieee_add(T.ieee_add(acc[0], acc[1]), acc[2]), acc[3])
            s = T.ieee_add(s, T.shfl_down(s, 16))
            s = T.ieee_add(s, T.shfl_down(s, 8))
            s = T.ieee_add(s, T.shfl_down(s, 4))
            s = T.ieee_add(s, T.shfl_down(s, 2))
            s = T.ieee_add(s, T.shfl_down(s, 1))
            s = T.shfl_sync(s, 0)
            r = T.rsqrt(T.ieee_add(T.ieee_mul(s, T.float32(1.0 / H)), T.float32(EPS)))   # torch: sum * f32(1/H)
            for c in T.serial(nch):
                for i in T.vectorized(4):
                    Y[row, (lane + 32 * c) * 4 + i] = T.cast(T.ieee_mul(
                        T.ieee_mul(T.cast(xl[c * 4 + i], F32), r),
                        T.ieee_add(T.float32(1.0), T.cast(gl[c * 4 + i], F32))), dt)

    return _compile(main, f"rmsnorm_{M}x{H}")


@lru_cache(maxsize=None)
def embed_kernel(M: int, V: int, H: int, threads: int = 128):
    """One CTA per token: the table row is staged through shared memory (as
    tir/small/embed_T*.py: an indexed copy into smem, then a copy out)."""
    dt = "bfloat16"

    @T.prim_func
    def main(ids: T.Tensor((M,), "int64"), table: T.Tensor((V, H), dt), Y: T.Tensor((M, H), dt)):
        with T.Kernel(M, threads=threads) as row:
            rs = T.alloc_shared((H,), dt)
            tok = ids[row]
            T.copy(table[tok, 0:H], rs)
            T.copy(rs, Y[row, 0:H])

    return _compile(main, f"embed_{M}")


@lru_cache(maxsize=None)
def gemv_kernel(N: int, K: int, rows: int = 16, threads: int = 256, vec: int = 8):
    """y = W x for one activation row. Each warp owns rows/warps rows; each lane
    streams 16-byte vectors of a row; the 32 partial sums meet in a warp reduce."""
    dt, acc = "bfloat16", "float"
    warps = threads // 32
    rows_per_warp = rows // warps
    chunks = K // (32 * vec)
    assert chunks * 32 * vec == K and rows_per_warp * warps == rows and N % rows == 0

    @T.prim_func
    def main(X: T.Tensor((K,), dt), W: T.Tensor((N, K), dt), Y: T.Tensor((N,), dt)):
        with T.Kernel(N // rows, threads=threads) as bx:
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            xs = T.alloc_shared((K,), dt)
            wl = T.alloc_local((vec,), dt)
            part = T.alloc_local((1,), acc)
            for i in T.Parallel(K):
                xs[i] = X[i]
            for r in T.serial(rows_per_warp):
                n = bx * rows + warp * rows_per_warp + r
                part[0] = T.float32(0)
                for c in T.serial(chunks):
                    base = (c * 32 + lane) * vec
                    for v in T.vectorized(vec):
                        wl[v] = W[n, base + v]
                    for v in T.serial(vec):
                        part[0] += T.cast(wl[v], acc) * T.cast(xs[base + v], acc)
                part[0] = part[0] + T.shfl_xor(part[0], 16)
                part[0] = part[0] + T.shfl_xor(part[0], 8)
                part[0] = part[0] + T.shfl_xor(part[0], 4)
                part[0] = part[0] + T.shfl_xor(part[0], 2)
                part[0] = part[0] + T.shfl_xor(part[0], 1)
                if lane == 0:
                    Y[n] = T.cast(part[0], dt)

    return _compile(main, f"gemv_{N}x{K}")


@lru_cache(maxsize=None)
def conv_silu_kernel(M: int, C: int, KW: int = 4, block: int = 256):
    """y[t, c] = bf16(silu(bf16(((x[t-3]w0 + x[t-2]w1) + x[t-1]w2) + x[t]w3)))  (HIR causal_conv)."""
    dt, acc = "bfloat16", "float"

    @T.prim_func
    def main(X: T.Tensor((M, C), dt), Wc: T.Tensor((C, KW), dt), Y: T.Tensor((M, C), dt)):
        with T.Kernel(M, T.ceildiv(C, block), threads=block) as (t, cb):
            tx = T.get_thread_binding()
            c = cb * block + tx
            s = T.alloc_local((1,), acc)
            s[0] = T.float32(0)
            for j in T.serial(KW):
                src = t + j - (KW - 1)
                if src >= 0:
                    s[0] = s[0] + T.cast(X[src, c], acc) * T.cast(Wc[c, j], acc)
            b = T.cast(T.cast(s[0], dt), acc)
            Y[t, c] = T.cast(b / (T.float32(1.0) + T.exp(-b)), dt)

    return _compile(main, f"conv_silu_{M}")


@lru_cache(maxsize=None)
def l2norm_kernel(M: int, NH: int, Dh: int = 128):
    """Per (t, head): bf16 arithmetic as HIR l2_normalise: s = bf16(sum(bf16(x*x)));
    r = bf16(rsqrt(bf16(s + bf16(1e-6)))); y = bf16(x * r). One warp per (t, head)."""
    dt, acc = "bfloat16", "float"
    per = Dh // 32

    @T.prim_func
    def main(X: T.Tensor((M, NH, Dh), dt), Y: T.Tensor((M, NH, Dh), dt)):
        with T.Kernel(M, NH, threads=32) as (t, h):
            lane = T.get_thread_binding()
            xl = T.alloc_local((per,), dt)
            s = T.alloc_local((1,), acc)
            for v in T.vectorized(per):
                xl[v] = X[t, h, lane * per + v]
            s[0] = T.float32(0)
            for v in T.serial(per):
                s[0] = s[0] + T.cast(T.cast(T.cast(xl[v], acc) * T.cast(xl[v], acc), dt), acc)
            s[0] = s[0] + T.shfl_xor(s[0], 16)
            s[0] = s[0] + T.shfl_xor(s[0], 8)
            s[0] = s[0] + T.shfl_xor(s[0], 4)
            s[0] = s[0] + T.shfl_xor(s[0], 2)
            s[0] = s[0] + T.shfl_xor(s[0], 1)
            ss = T.cast(T.cast(s[0], dt), acc)
            e = T.cast(T.cast(T.float32(EPS), dt), acc)
            r = T.cast(T.cast(T.rsqrt(T.cast(T.cast(ss + e, dt), acc)), dt), acc)
            for v in T.serial(per):
                Y[t, h, lane * per + v] = T.cast(T.cast(xl[v], acc) * r, dt)

    return _compile(main, f"l2norm_{M}x{NH}")


@lru_cache(maxsize=None)
def rope_kernel(M: int, NH: int, Dh: int = 256, R: int = 64):
    """HIR partial_rope: first R dims rotate-half in bf16 arithmetic, the rest copied."""
    dt, acc = "bfloat16", "float"
    half = R // 2

    @T.prim_func
    def main(X: T.Tensor((M, NH, Dh), dt), Cs: T.Tensor((M, R), dt), Sn: T.Tensor((M, R), dt), Y: T.Tensor((M, NH, Dh), dt)):
        with T.Kernel(M, NH, threads=Dh) as (t, h):
            d = T.get_thread_binding()
            if d < R:
                x = T.cast(X[t, h, d], acc)
                partner = T.if_then_else(d < half, -T.cast(X[t, h, d + half], acc), T.cast(X[t, h, d - half], acc))
                a = T.cast(T.cast(x * T.cast(Cs[t, d], acc), dt), acc)
                b = T.cast(T.cast(partner * T.cast(Sn[t, d], acc), dt), acc)
                Y[t, h, d] = T.cast(a + b, dt)
            else:
                Y[t, h, d] = X[t, h, d]

    return _compile(main, f"rope_{M}x{NH}")
