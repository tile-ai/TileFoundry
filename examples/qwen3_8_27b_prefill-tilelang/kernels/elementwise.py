"""The memory-bound prefill kernels: row norms, the GDN prep and gated norm, the gather.

Each lands where model.py's function lands (the published classes' order):

  rmsnorm     Qwen3_5RMSNorm: y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)), f32 inside
  gated_norm  Qwen3_5RMSNormGated: n = bf16(x * rsqrt(mean(x^2) + eps));
              y = bf16(f32(bf16(n * w)) * silu(f32(z)))
  prep        causal depthwise conv (f32, taps t, t-1, t-2, t-3), bf16, SiLU in
              bf16; q and k heads l2-normalised at bf16 (FLA's l2norm as
              transformers runs it); g = -exp(A_log) softplus(a + dt_bias) f32,
              beta = sigmoid(b) bf16
  embed       one table row per token
"""

import tilelang
import tilelang.language as T

BF16, F32 = "bfloat16", "float32"


def rmsnorm_kernel(M: int, H: int = 5120, rows: int = 2, threads: int = 256, eps: float = 1e-6, VEC: int = 4):
    """tir/rmsnorm_{M}.py (scheduled/norm.py): M / 2 CTAs x 256 threads; all
    256 threads span both rows of the pair: thread t holds, of each row, the
    VEC-wide vectors at columns (k * 256 + t) * VEC + j (k < H / (256 VEC)),
    gamma likewise. The mean of squares: per thread, a 32-lane butterfly, then
    the eight warp sums from shared memory in warp order (T.reduce's
    workspace)."""
    TV = threads
    KV = H // (TV * VEC)
    NW = TV // 32

    @T.prim_func
    def rmsnorm(X: T.Tensor((M, H), BF16), G: T.Tensor((H,), BF16), Y: T.Tensor((M, H), BF16)):
        with T.Kernel(M // rows, threads=threads) as bx:
            t = T.get_thread_binding()
            xr = T.alloc_local((rows * KV * VEC,), BF16)
            gr = T.alloc_local((KV * VEC,), BF16)
            yb = T.alloc_local((VEC,), BF16)
            red = T.alloc_local((rows,), F32)
            ws = T.alloc_shared((rows * NW,), F32)
            for r in T.unroll(rows):
                for k in T.unroll(KV):
                    for j in T.vectorized(VEC):
                        xr[(r * KV + k) * VEC + j] = X[bx * rows + r, (k * TV + t) * VEC + j]
            for r in T.unroll(rows):
                red[r] = 0.0
                for i in T.unroll(KV * VEC):
                    red[r] = red[r] + T.cast(xr[r * KV * VEC + i], F32) * T.cast(xr[r * KV * VEC + i], F32)
                for o in T.unroll(5):
                    red[r] = red[r] + T.shfl_xor(red[r], 16 >> o, 32)
                if t % 32 == 0:
                    ws[r * NW + t // 32] = red[r]
            T.sync_threads()
            for k in T.unroll(KV):
                for j in T.vectorized(VEC):
                    gr[k * VEC + j] = G[(k * TV + t) * VEC + j]
            for r in T.unroll(rows):
                red[r] = 0.0
                for w in T.unroll(NW):
                    red[r] = red[r] + ws[r * NW + w]
                rs = T.rsqrt(red[r] / H + eps)
                for k in T.unroll(KV):
                    for j in T.unroll(VEC):
                        yb[j] = T.cast(T.cast(xr[(r * KV + k) * VEC + j], F32) * rs * (T.cast(gr[k * VEC + j], F32) + 1.0), BF16)
                    for j in T.vectorized(VEC):
                        Y[bx * rows + r, (k * TV + t) * VEC + j] = yb[j]

    return rmsnorm


def gated_norm_kernel(M: int, HV: int = 48, DV: int = 128, ZS: int = 16480, Z0: int = 10240,
                      RPT: int = 1, eps: float = 1e-6):
    """core (M * HV, DV) bf16, one row per (token, value head); z read from the
    in_proj output (row stride ZS, column Z0). 16 threads cover one row, 8
    columns (16 bytes) each; RMS of the row by a 16-lane shuffle butterfly;
    each thread does RPT rows. Model.py's order: n = bf16(x rsqrt(mean x^2 +
    eps)), bf16(n gamma), times silu(z) in f32, landing once."""
    ROWS = M * HV
    RB = 16 * RPT   # rows per CTA (256 threads = 16 row slots x 16 lanes)

    @T.prim_func
    def gated_norm(Core: T.Tensor((ROWS, DV), BF16), Z: T.Tensor((M, ZS), BF16), G: T.Tensor((DV,), BF16),
                   Y: T.Tensor((ROWS, DV), BF16)):
        with T.Kernel(ROWS // RB, threads=256) as bx:
            tx = T.get_thread_binding()
            lane = tx % 16
            xr = T.alloc_local((RPT * 8,), BF16)
            zr = T.alloc_local((RPT * 8,), BF16)
            gr = T.alloc_local((8,), BF16)
            yb = T.alloc_local((8,), BF16)
            red = T.alloc_local((1,), F32)
            for j in T.vectorized(8):
                gr[j] = G[lane * 8 + j]
            for r in T.unroll(RPT):
                row = bx * RB + r * 16 + tx // 16
                for j in T.vectorized(8):
                    xr[r * 8 + j] = Core[row, lane * 8 + j]
                for j in T.vectorized(8):
                    zr[r * 8 + j] = Z[row // HV, Z0 + (row % HV) * DV + lane * 8 + j]
            for r in T.unroll(RPT):
                row = bx * RB + r * 16 + tx // 16
                red[0] = 0.0
                for j in T.unroll(8):
                    red[0] = red[0] + T.cast(xr[r * 8 + j], F32) * T.cast(xr[r * 8 + j], F32)
                for o in T.unroll(4):
                    red[0] = red[0] + T.shfl_xor(red[0], 8 >> o, 16)
                rs = T.rsqrt(red[0] / DV + eps)
                for j in T.unroll(8):
                    n = T.cast(T.cast(xr[r * 8 + j], F32) * rs, BF16)
                    nw = T.cast(T.cast(n, F32) * T.cast(gr[j], F32), BF16)
                    zz = T.cast(zr[r * 8 + j], F32)
                    yb[j] = T.cast(T.cast(nw, F32) * T.call_extern(F32, "__fdividef", zz, 1.0 + T.call_extern(F32, "__expf", -zz)), BF16)
                for j in T.vectorized(8):
                    Y[row, lane * 8 + j] = yb[j]

    return gated_norm


def prep_kernel(M: int, R: int = 8, S: int = 8, QS: int = 16480, eps_l2: float = 1e-6):
    """tir/prep_{conv_q,conv_k,conv_v,gate_g,gate_beta}_{M}.py (scheduled/gdn_elementwise.py),
    the five device functions of model.py's prep, in one launch.

    qkv / b / a are read from the in_proj output (row stride QS): qkv at
    columns [0, 10240), b at [16384, 16432), a at [16432, 16480).

    Grid (M / 64, 80) x 128 threads as (s, g) = (8, 16). CTA (bt, cb): rows
    64 bt .., channel block cb (q heads 0-15, k 16-31, v 32-79). The 3 rows
    before the block (zero for the first block) and its 64 rows are staged in
    shared memory as a (67, 128) sm0_slab; thread (s, g) owns rows 8 s .. 8 s + 7
    and the 16-byte channel vector g, and reads tap 3 - sh of row t from sm0_slab
    row t + 3 - sh. f32 sum in model.py's tap order, bf16, SiLU, and for q / k
    the l2norm with every step in bf16 (row sum: 16-lane butterfly). The
    cb = 0 CTAs also compute the gates of their 64 rows: thread (s, g) rows
    8 s .., heads 3 g .. 3 g + 2."""
    CONV, HK, HV, D = 10240, 16, 48, 128
    B0, A0 = 16384, 16432
    NB = CONV // D  # 80 channel blocks: 16 q heads, 16 k heads, 48 v heads
    BT = R * S
    NBT = M // BT
    HE = HV // 16   # gate heads per thread

    @T.prim_func
    def prep(QKV: T.Tensor((M, QS), BF16), CW: T.Tensor((CONV, 4), BF16),
             ALOG: T.Tensor((HV,), BF16), DTB: T.Tensor((HV,), BF16),
             Q: T.Tensor((M, HK, D), BF16), Kk: T.Tensor((M, HK, D), BF16), V: T.Tensor((M, HV, D), BF16),
             Gg: T.Tensor((M, HV), F32), Beta: T.Tensor((M, HV), BF16)):
        with T.Kernel(NBT, NB, threads=16 * S) as (bt, cb):
            tx = T.get_thread_binding(0)
            sm0_slab = T.alloc_shared((BT + 3, D), BF16)
            w = T.alloc_local((32,), BF16)
            xv = T.alloc_local((8,), BF16)
            xt = T.alloc_local((32,), BF16)
            wf = T.alloc_local((32,), F32)
            yb = T.alloc_local((8,), BF16)
            yf = T.alloc_local((8,), F32)
            red = T.alloc_local((1,), F32)
            g = tx % 16
            s = tx // 16
            m0 = bt * BT
            ch = cb * D + g * 8
            keep = T.if_then_else(bt > 0, T.cast(1.0, BF16), T.cast(0.0, BF16))
            hs = (m0 + M - 3) % M
            for i in T.unroll(3):
                for j in T.vectorized(8):
                    xv[j] = QKV[hs + i, ch + j]
                for j in T.vectorized(8):
                    sm0_slab[i, g * 8 + j] = xv[j] * keep
            for i in T.unroll(R):
                for j in T.vectorized(8):
                    sm0_slab[3 + s * R + i, g * 8 + j] = QKV[m0 + s * R + i, ch + j]
            for j in T.vectorized(32):
                w[j] = CW[ch + j // 4, j % 4]
            for j in T.unroll(32):
                wf[j] = T.cast(w[j], F32)
            T.sync_threads()
            for i in T.unroll(R):
                row = 3 + s * R + i
                for sh in T.unroll(4):
                    for j in T.vectorized(8):
                        xt[sh * 8 + j] = sm0_slab[row - sh, g * 8 + j]
                for j in T.unroll(8):
                    acc = T.cast(xt[j], F32) * wf[j * 4 + 3]
                    acc = acc + T.cast(xt[8 + j], F32) * wf[j * 4 + 2]
                    acc = acc + T.cast(xt[16 + j], F32) * wf[j * 4 + 1]
                    acc = acc + T.cast(xt[24 + j], F32) * wf[j * 4]
                    xb = T.cast(T.cast(acc, BF16), F32)
                    yf[j] = T.cast(T.cast(T.call_extern(F32, "__fdividef", xb, 1.0 + T.call_extern(F32, "__expf", -xb)), BF16), F32)
                if cb < 2 * HK:
                    # l2norm at bf16: every product, the sum, the eps add, the
                    # rsqrt and the scaling each land in bf16.
                    red[0] = 0.0
                    for j in T.unroll(8):
                        red[0] = red[0] + T.cast(T.cast(yf[j] * yf[j], BF16), F32)
                    for o in T.unroll(4):
                        red[0] = red[0] + T.shfl_xor(red[0], 8 >> o, 16)
                    sc = T.cast(T.cast(red[0], BF16), F32)
                    sc = T.cast(T.cast(sc + eps_l2, BF16), F32)
                    sc = T.cast(T.cast(T.rsqrt(sc), BF16), F32)
                    for j in T.unroll(8):
                        yb[j] = T.cast(yf[j] * sc, BF16)
                    if cb < HK:
                        for j in T.vectorized(8):
                            Q[m0 + s * R + i, cb, g * 8 + j] = yb[j]
                    else:
                        for j in T.vectorized(8):
                            Kk[m0 + s * R + i, cb - HK, g * 8 + j] = yb[j]
                else:
                    for j in T.unroll(8):
                        yb[j] = T.cast(yf[j], BF16)
                    for j in T.vectorized(8):
                        V[m0 + s * R + i, cb - 2 * HK, g * 8 + j] = yb[j]
            if cb == 0:
                for i in T.unroll(R):
                    for e in T.unroll(HE):
                        r = m0 + s * R + i
                        h = g * HE + e
                        bb = T.cast(QKV[r, B0 + h], F32)
                        Beta[r, h] = T.cast(1.0 / (1.0 + T.exp(-bb)), BF16)
                        x = T.cast(QKV[r, A0 + h], F32) + T.cast(DTB[h], F32)
                        u = T.exp(x)
                        v1 = u + 1.0
                        lp = T.if_then_else(v1 == 1.0, u, T.log(v1) * u / (v1 - 1.0))
                        sp = T.if_then_else(x > 20.0, x, lp)
                        Gg[r, h] = -T.exp(T.cast(ALOG[h], F32)) * sp

    return prep


def embed_kernel(M: int, V: int = 248320, H: int = 5120, rows: int = 4, VEC: int = 4):
    """tir/embed_{M}.py (scheduled/embed_final.py): M / 4 CTAs x 256 threads;
    for each of its 4 rows the CTA reads the token id and thread t moves the
    8-byte vectors at columns (k * 256 + t) * 4 of the table row."""
    TV = 256
    KV = H // (TV * VEC)

    @T.prim_func
    def embed(Ids: T.Tensor((M,), "int64"), Table: T.Tensor((V, H), BF16), Y: T.Tensor((M, H), BF16)):
        with T.Kernel(M // rows, threads=TV) as bx:
            t = T.get_thread_binding()
            v = T.alloc_local((VEC,), BF16)
            for i in T.unroll(rows):
                tok = Ids[bx * rows + i]
                for k in T.unroll(KV):
                    for j in T.vectorized(VEC):
                        v[j] = Table[tok, (k * TV + t) * VEC + j]
                    for j in T.vectorized(VEC):
                        Y[bx * rows + i, (k * TV + t) * VEC + j] = v[j]

    return embed


_CACHE = {}


def compiled(name, *args, **kw):
    key = (name, args, tuple(sorted(kw.items())))
    if key not in _CACHE:
        fn = {"rmsnorm": rmsnorm_kernel, "gated_norm": gated_norm_kernel, "prep": prep_kernel,
              "embed": embed_kernel}[name]
        _CACHE[key] = tilelang.compile(fn(*args, **kw), out_idx=None,
                                       pass_configs={"tl.disable_shared_memory_reuse": True})
    return _CACHE[key]
