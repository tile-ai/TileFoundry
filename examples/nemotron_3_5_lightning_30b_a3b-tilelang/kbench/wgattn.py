"""One attention layer's block loop, on the tensor cores, end to end.

This is the shape that goes into the mega kernel: both products are WGMMA, the
online-softmax merge happens on the accumulator fragment between them, and the
K/V block is one bulk-copy-shaped (positions x both KV heads) buffer that each
group reads a column slice of -- no restride, no staging half.
"""
import time
import torch
import tilelang
import tilelang.language as T

HQ, HKV, DH, ABLK = 32, 2, 128, 128
GQA = HQ // HKV
KVROW = HKV * DH
QSCALE = DH ** -0.5
NEG = -3.0e38
THREADS = 256
NBLK = 2


@tilelang.jit
def build():
    @T.prim_func
    def main(q: T.Tensor((HQ, DH), "float32"),
             kv: T.Tensor((NBLK, ABLK, KVROW), "bfloat16"),
             vv: T.Tensor((NBLK, ABLK, KVROW), "bfloat16"),
             nvalid: T.Tensor((1,), "int32"),
             out: T.Tensor((HQ, DH), "float32")):
        with T.Kernel(1, threads=THREADS) as bx:
            qs = T.alloc_shared((HQ, DH), "bfloat16")
            ksm = T.alloc_shared((ABLK, KVROW), "bfloat16")
            vsm = T.alloc_shared((ABLK, KVROW), "bfloat16")
            ss = T.alloc_shared((GQA, ABLK), "float32")
            ps = T.alloc_shared((GQA, ABLK), "bfloat16")
            mv = T.alloc_shared((HKV * GQA,), "float32")
            lv = T.alloc_shared((HKV * GQA,), "float32")
            cr = T.alloc_shared((HKV * GQA,), "float32")
            sf = T.alloc_fragment((GQA, ABLK), "float32")
            # One accumulator per KV head. A gemm's output layout is built for two
            # dimensions, and a slice of a three-dimensional fragment is not one.
            of0 = T.alloc_fragment((GQA, DH), "float32")
            of1 = T.alloc_fragment((GQA, DH), "float32")
            for i, j in T.Parallel(HQ, DH):
                qs[i, j] = T.Cast("bfloat16", q[i, j] * QSCALE)
            T.clear(of0)
            T.clear(of1)
            for i in T.Parallel(HKV * GQA):
                mv[i] = NEG
                lv[i] = 0.0
            for b in T.serial(NBLK):
                T.copy(kv[b, :, :], ksm)
                T.copy(vv[b, :, :], vsm)
                T.clear(sf)
                # `T.gemm`, not `T.wgmma_gemm`: WGMMA's M is fixed at 64 and GQA
                # puts 16 query heads on one KV head, so the warpgroup form runs
                # three quarters empty. At M=16 the tensor-core instruction that
                # fits is `mma.m16n8k16`, which is what this selects.
                T.gemm(qs[0 * GQA:1 * GQA, :],
                       ksm[:, 0 * DH:1 * DH], sf, transpose_B=True)
                # The scores go to shared to be normalised. Reducing along the
                # accumulator's N axis in place cannot be lowered at M=16: eight
                # warps split N, so a row's 128 values are spread across all of
                # them and the reduce is not projectable onto one thread's own
                # segment. Shared costs 8 KB out and 8 KB back per block.
                T.copy(sf, ss)
                for i in T.Parallel(GQA):
                    ss[i, 0] = ss[i, 0]
                for i in T.serial(GQA):
                    cr[0 * GQA + i] = 0.0
                for i in T.Parallel(GQA):
                    bmx = T.alloc_var("float32")
                    bmx = NEG
                    for j in T.serial(ABLK):
                        bmx = T.max(bmx, T.if_then_else(b * ABLK + j < nvalid[0],
                                                        ss[i, j], NEG))
                    cr[0 * GQA + i] = T.exp(mv[0 * GQA + i]
                                              - T.max(mv[0 * GQA + i], bmx))
                    mv[0 * GQA + i] = T.max(mv[0 * GQA + i], bmx)
                    bsum = T.alloc_var("float32")
                    bsum = 0.0
                    for j in T.serial(ABLK):
                        ss[i, j] = T.if_then_else(
                            b * ABLK + j < nvalid[0],
                            T.exp(ss[i, j] - mv[0 * GQA + i]), 0.0)
                        bsum = bsum + ss[i, j]
                    lv[0 * GQA + i] = lv[0 * GQA + i] * cr[0 * GQA + i] + bsum
                for i, j in T.Parallel(GQA, ABLK):
                    ps[i, j] = T.Cast("bfloat16", ss[i, j])
                for i, j in T.Parallel(GQA, DH):
                    of0[i, j] = of0[i, j] * cr[0 * GQA + i]
                T.gemm(ps, vsm[:, 0 * DH:1 * DH], of0)
                T.clear(sf)
                # `T.gemm`, not `T.wgmma_gemm`: WGMMA's M is fixed at 64 and GQA
                # puts 16 query heads on one KV head, so the warpgroup form runs
                # three quarters empty. At M=16 the tensor-core instruction that
                # fits is `mma.m16n8k16`, which is what this selects.
                T.gemm(qs[1 * GQA:2 * GQA, :],
                       ksm[:, 1 * DH:2 * DH], sf, transpose_B=True)
                # The scores go to shared to be normalised. Reducing along the
                # accumulator's N axis in place cannot be lowered at M=16: eight
                # warps split N, so a row's 128 values are spread across all of
                # them and the reduce is not projectable onto one thread's own
                # segment. Shared costs 8 KB out and 8 KB back per block.
                T.copy(sf, ss)
                for i in T.Parallel(GQA):
                    ss[i, 0] = ss[i, 0]
                for i in T.serial(GQA):
                    cr[1 * GQA + i] = 0.0
                for i in T.Parallel(GQA):
                    bmx = T.alloc_var("float32")
                    bmx = NEG
                    for j in T.serial(ABLK):
                        bmx = T.max(bmx, T.if_then_else(b * ABLK + j < nvalid[0],
                                                        ss[i, j], NEG))
                    cr[1 * GQA + i] = T.exp(mv[1 * GQA + i]
                                              - T.max(mv[1 * GQA + i], bmx))
                    mv[1 * GQA + i] = T.max(mv[1 * GQA + i], bmx)
                    bsum = T.alloc_var("float32")
                    bsum = 0.0
                    for j in T.serial(ABLK):
                        ss[i, j] = T.if_then_else(
                            b * ABLK + j < nvalid[0],
                            T.exp(ss[i, j] - mv[1 * GQA + i]), 0.0)
                        bsum = bsum + ss[i, j]
                    lv[1 * GQA + i] = lv[1 * GQA + i] * cr[1 * GQA + i] + bsum
                for i, j in T.Parallel(GQA, ABLK):
                    ps[i, j] = T.Cast("bfloat16", ss[i, j])
                for i, j in T.Parallel(GQA, DH):
                    of1[i, j] = of1[i, j] * cr[1 * GQA + i]
                T.gemm(ps, vsm[:, 1 * DH:2 * DH], of1)
            for i, j in T.Parallel(GQA, DH):
                out[i, j] = of0[i, j] / lv[i]
            for i, j in T.Parallel(GQA, DH):
                out[GQA + i, j] = of1[i, j] / lv[GQA + i]
    return main


def reference(q, kv, vv, n):
    q = q.float() * QSCALE
    k = kv.float().reshape(NBLK * ABLK, KVROW)[:n]
    v = vv.float().reshape(NBLK * ABLK, KVROW)[:n]
    out = torch.zeros(HQ, DH, device=q.device)
    for h in range(HQ):
        g = h // GQA
        s = q[h] @ k[:, g * DH:(g + 1) * DH].t()
        w = torch.softmax(s, dim=-1).to(torch.bfloat16).float()
        out[h] = w @ v[:, g * DH:(g + 1) * DH]
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(HQ, DH, device="cuda")
    kv = torch.randn(NBLK, ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    vv = torch.randn(NBLK, ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(HQ, DH, device="cuda")
    fn = build()
    for n in (NBLK * ABLK, 200, 129, 7):
        nv = torch.tensor([n], device="cuda", dtype=torch.int32)
        out.zero_()
        fn(q, kv, vv, nv, out)
        torch.cuda.synchronize()
        ref = reference(q, kv, vv, n)
        print(f"n={n:4d}  rel_l2 {((out - ref).norm() / ref.norm()).item():.3e}"
              f"  max_abs {(out - ref).abs().max().item():.3e}")
