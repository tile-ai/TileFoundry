"""Scheduled HIR for model.py's `Attention.attention`: causal flash attention
with the sigmoid output gate, FA3-style on H200.

One Module per sequence length; `schedule finalize` lowers `attention` to
tir/attention_{n}.py, implemented by kernels/attention.py `attention_tir_kernel`.

  grid      (n / 128 query blocks, 24 heads); query block qb runs block
            n/128 - 1 - qb (the longest rows first). 384 threads: warp 0 of
            warpgroup 2 loads Q once and K, V blocks of 80 keys by TMA through
            2-deep rings (slot kb mod 2); warpgroups 0 and 1 each own 64 query
            rows.
  per key block: S = Q K^T (m64n80k16 WGMMA, both K-major SW128); on a block
            the diagonal crosses (kb >= q0 // 80, a guard loop of 0 or 1
            iterations) the causal mask, row - column numbers from the iota
            table (i32); the online softmax in f32 (row max / sum over the
            fragment's row, exp2 with the 1/sqrt(d) log2(e) scale folded in),
            O *= the correction, O += P V (five m64n256k16 RS WGMMAs: P from
            registers in bf16, V MN-major in smem as five 16-key slabs of
            4 x 64 columns), the key-block loop bound by the causal limit.
            The runtime issues them in this order; its two warpgroups take
            turns at the tensor cores (FA3's ping-pong between warpgroups).
  epilogue  O / l landing in bf16, times bf16(sigmoid(gate)), landing again;
            staged in SW128 smem (Q's place) and stored by TMA.
  constants an integer literal is i64 and never promoted: the i32 zero and the
            i32 offsets come from register tensors (tf.zeros(..., "rmem")).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
HQ, HKV, D = 24, 4, 256
GQA = HQ // HKV
QKVW = HQ * 2 * D + 2 * HKV * D
K0, V0 = HQ * 2 * D, HQ * 2 * D + HKV * D
BM, BN = 128, 80
SCALE = (D ** -0.5) * 1.4426950408889634


def attention(M: int):
    NQB = M // BM
    KBMOD = (M + BN - 1) // BN + 2   # above every key-block index: (x + 0) % KBMOD = x, a bounded form

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NQB * HQ), Topology("thread", 384)))
    class Attention:
        @func
        def attention(qkv: Tensor[(M, QKVW), "bf16"], iota: Tensor[(BM,), "i32"]):
            with Mesh(("cta",), layout=(NQB, HQ), names=("qb", "h")) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    wqk = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    wpv = T.cuda.sm90.Wgmma(n=D, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                    out = tf.zeros(Tensor[(M, HQ * D), "bf16"])
                    q0 = (NQB - 1 - cta.qb) * BM
                    g = cta.h // GQA
                    with threads[2, :32] as _loader:
                        qs = tf.schedule((qkv[q0:q0 + BM, cta.h * 2 * D:cta.h * 2 * D + D],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3)))
                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                        o = tf.zeros(Tensor[(BM, D), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                        mx = tf.zeros(Tensor[(BM, 1), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0)), "rmem"]) - 1e30
                        rows = tf.schedule((tf.reshape(iota, (BM, 1)),), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0)))) + tf.cast(tf.zeros(Tensor[(1, 1), "i64", (1, 1), "rmem"]) + q0, "i32")
                        lsum = tf.zeros(Tensor[(BM, 1), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0)), "rmem"])
                    for kb in range(((NQB - 1 - cta.qb) * BM + BM + BN - 1) // BN):
                        k0 = kb * BN
                        with threads[2, :32] as _loader:
                            ks = tf.schedule((qkv[k0:k0 + BN, K0 + g * D:K0 + g * D + D],), op=T.copy_async_tensor(smem_layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3)), buffers=2)
                            vs = tf.schedule((qkv[k0:k0 + BN, V0 + g * D:V0 + g * D + D],), op=T.copy_async_tensor(smem_layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3)), buffers=2)
                        with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                            s0 = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 10), (5120, 1, 8, 16, 64, 128, 512)), "rmem"])
                            s = tf.schedule((s0, qs, tf.bitcast(ks, layout=Layout(((4, 4, 16), (10, 8)), ((5120, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=wqk), repeat=(2, 1, 16))
                            for _dg in range(((NQB - 1 - cta.qb) * BM) // BN - 1, (cta.qb * 0 + kb) % KBMOD, 1048576):   # once for a block the diagonal crosses (kb >= q0 // BN)
                                cols = tf.schedule((tf.reshape(iota[0:BN], (1, BN)),), op=T.copy(rmem_layout=((1, 2, 4 @ c.lane4, 10), (0, 64, 128, 512)))) + tf.cast(tf.zeros(Tensor[(1, 1), "i64", (1, 1), "rmem"]) + k0, "i32")
                                dif = tf.schedule((rows - cols,), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 10), (5120, 1, 8, 16, 64, 128, 512))))
                                s = tf.where(dif >= tf.zeros(Tensor[(1, 1), "i32", (1, 1), "rmem"]), s, -1e30)
                            mnew = tf.maximum(mx, tf.schedule((tf.reduce(s, (1,), True, ReduceKind.MAX),), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0)))))
                            corr = tf.exp2(mx * SCALE - mnew * SCALE)
                            p = tf.exp2(s * SCALE - mnew * SCALE)
                            lsum = lsum * corr + tf.schedule((tf.reduce(p, (1,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))))
                            o = o * corr
                            pb = tf.schedule((tf.cast(p, "bf16"),), op=T.copy(rmem_layout=((2 @ c.group, 5, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 2), (5120, 1024, 1, 8, 16, 64, 128, 512))))
                            o = tf.schedule((o, pb, tf.bitcast(vs, layout=Layout(((5, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=wpv), repeat=(2, 1, 5))
                            mx = mnew
                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                        ob = tf.cast(tf.cast(o / lsum, "bf16"), "f32")
                        gt = tf.cast(tf.schedule((qkv[q0:q0 + BM, cta.h * 2 * D + D:cta.h * 2 * D + 2 * D],), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)))), "f32")
                        sg = tf.cast(tf.cast(1.0 / (tf.exp(-gt) + 1.0), "bf16"), "f32")
                        y = tf.cast(ob * sg, "bf16")
                        staged = tf.schedule((y,), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                    with threads[0, :32] as _storer:
                        tile = tf.schedule((staged,), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (q0, cta.h * D))
                    return out

        @func
        def entry(q: Tensor[(1, M, HQ, D), "bf16"], k: Tensor[(1, M, HKV, D), "bf16"], v: Tensor[(1, M, HKV, D), "bf16"],
                  gate: Tensor[(1, M, HQ, D), "bf16"]) -> Tensor[(1, M, HQ * D), "bf16"]:
            # the kernel reads the qkv GEMM's packed output [q|gate per head | k | v] as it lies,
            # and the row / column numbers of its causal mask from a table (TIR has no index op)
            qg = tf.reshape(tf.concat([q, gate], axis=-1), (M, HQ * 2 * D))
            qkv = tf.concat([qg, tf.reshape(k, (M, HKV * D)), tf.reshape(v, (M, HKV * D))], axis=1)
            iota = tf.arange(type=Tensor[(BM,), "i32"], start=0, step=1)
            return tf.reshape(attention(qkv, iota), (1, M, HQ * D))  # noqa: F821

    Attention.__doc__ = f"attention at seq_len={M}: ({NQB}, {HQ}) CTAs x 384 threads"
    return Attention


for _M in (1024, 4096, 8192):
    globals()[f"Attention{_M}"] = attention(_M)
