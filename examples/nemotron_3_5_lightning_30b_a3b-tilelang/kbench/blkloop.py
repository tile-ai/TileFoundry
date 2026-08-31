"""The attention block loop as it really runs: streamed operands, honest cost.

A gemm whose operands never change lets the compiler hoist every shared-to-
register load out of the timing loop and leaves only the tensor-core
instructions being counted.  Here each block's K and V come from memory, so the
operand loads are inside the measurement -- which is where a shared layout the
tensor cores did not choose gets to show what it costs.

    linear    the bulk copy's own bytes: `T.tma_copy` into the flat view of a
              two-dimensional buffer pinned to a linear layout
    swizzled  the layout `T.gemm` picks, filled by a plain `T.copy` from the
              same table pointer -- descriptorless, so cp.async rather than TMA
"""
import argparse
import time
import torch
import tilelang
import tilelang.language as T

import os
HQ, HKV, GQA, DH = 32, 2, 16, 128
ABLK = int(os.environ.get('BL_ABLK', 128))
KVROW = HKV * DH
NEL = ABLK * KVROW
CTAS, THREADS = 132, 256


def make(layout, gemm, nb):
    linear = layout == "linear"
    single = layout == "single"
    col = layout == "col"

    @tilelang.jit
    def build():
        @T.prim_func
        def main(ptrs: T.Tensor((2,), "int64"), meta: T.Tensor((1,), "int32"),
                 q: T.Tensor((HQ, DH), "float32"), o: T.Tensor((HQ, DH), "float32")):
            with T.Kernel(CTAS, threads=THREADS) as bx:
                ks = T.alloc_shared((ABLK, DH if col else KVROW), "bfloat16")
                vs = T.alloc_shared((ABLK, DH if col else KVROW), "bfloat16")
                qs = T.alloc_shared((HQ, DH), "bfloat16")
                ps = T.alloc_shared((GQA, ABLK), "bfloat16")
                fo = T.alloc_shared((HQ, DH), "float32")
                sf = T.alloc_fragment((GQA, ABLK), "float32")
                of0 = T.alloc_fragment((GQA, DH), "float32")
                of1 = T.alloc_fragment((GQA, DH), "float32")
                pf0 = T.alloc_fragment((GQA, ABLK), "bfloat16")
                pf1 = T.alloc_fragment((GQA, ABLK), "bfloat16")
                st32 = T.alloc_shared((GQA, ABLK), "float32")
                bar = T.alloc_barrier([1] * 2)
                tid = T.get_thread_binding()
                if linear:
                    T.annotate_layout({
                        ks: T.Layout((ABLK, KVROW), lambda i, j: i * KVROW + j),
                        vs: T.Layout((ABLK, KVROW), lambda i, j: i * KVROW + j)})
                kf = T.view(ks, (NEL if not col else ABLK * DH,))
                vf = T.view(vs, (NEL if not col else ABLK * DH,))
                flat = T.make_tensor_from_addr(ptrs[0], (1 << 28,), "bfloat16")
                tile = T.make_tensor_from_addr(ptrs[0], (1 << 21, ABLK, KVROW), "bfloat16")
                st = T.Cast("int32", meta[0])
                for i, j in T.Parallel(HQ, DH):
                    qs[i, j] = T.Cast("bfloat16", q[i, j])
                T.clear(of0)
                T.clear(of1)
                for b in T.serial(nb):
                    # K then V, one block apart, from this CTA's own stripe.
                    if linear:
                        lok = T.max(T.min((bx * st + b) * 2 * NEL,
                                          (1 << 28) - NEL), 0)
                        T.tma_copy(flat[lok:lok + NEL], kf, barrier=bar[0])
                        if T.shuffle_elect(THREADS):
                            T.barrier_arrive(bar[0])
                        lov = T.max(T.min((bx * st + b) * 2 * NEL + NEL,
                                          (1 << 28) - NEL), 0)
                        T.tma_copy(flat[lov:lov + NEL], vf, barrier=bar[1])
                        if T.shuffle_elect(THREADS):
                            T.barrier_arrive(bar[1])
                        T.mbarrier_wait_parity(bar[0], 0)
                        T.mbarrier_wait_parity(bar[1], 0)
                        if T.shuffle_elect(THREADS):
                            T.barrier_arrive(bar[0])
                        T.mbarrier_wait_parity(bar[0], 1)
                        if T.shuffle_elect(THREADS):
                            T.barrier_arrive(bar[1])
                        T.mbarrier_wait_parity(bar[1], 1)
                    elif col:
                        # One KV group at a time: K and V are column slices of
                        # the block, so both fit and both are in flight at once.
                        ib = T.max(T.min((bx * st + b) * 2, (1 << 21) - 2), 0)
                        T.copy(tile[ib, :, 0:DH], ks, prefer_instruction="cp_async")
                        T.copy(tile[ib + 1, :, 0:DH], vs, prefer_instruction="cp_async")
                        T.sync_threads()
                        if gemm:
                            T.clear(sf)
                            T.gemm(qs[0:GQA, :], ks, sf, transpose_B=True)
                            T.copy(sf, ps)
                            T.sync_threads()
                            T.gemm(ps, vs, of0)
                        else:
                            for i, j in T.Parallel(GQA, DH):
                                of0[i, j] += T.Cast("float32", ks[i, j])
                        T.copy(tile[ib, :, DH:2 * DH], ks, prefer_instruction="cp_async")
                        T.copy(tile[ib + 1, :, DH:2 * DH], vs, prefer_instruction="cp_async")
                        T.sync_threads()
                        if gemm:
                            T.clear(sf)
                            T.gemm(qs[GQA:2 * GQA, :], ks, sf, transpose_B=True)
                            T.copy(sf, ps)
                            T.sync_threads()
                            T.gemm(ps, vs, of1)
                        else:
                            for i, j in T.Parallel(GQA, DH):
                                of1[i, j] += T.Cast("float32", vs[i, j])
                    elif single:
                        # One buffer: K for both groups' scores, then V over it.
                        ib = T.max(T.min((bx * st + b) * 2, (1 << 21) - 2), 0)
                        T.copy(tile[ib, :, :], ks, prefer_instruction="cp_async")
                        T.sync_threads()
                        if gemm:
                            T.clear(sf)
                            T.gemm(qs[0:GQA, :], ks[:, 0:DH], sf, transpose_B=True)
                            T.copy(sf, st32)
                            T.sync_threads()
                            T.copy(st32, pf0)
                            T.clear(sf)
                            T.gemm(qs[GQA:2 * GQA, :], ks[:, DH:2 * DH], sf,
                                   transpose_B=True)
                            T.copy(sf, st32)
                            T.sync_threads()
                            T.copy(st32, pf1)
                        else:
                            for i, j in T.Parallel(GQA, DH):
                                of0[i, j] += T.Cast("float32", ks[i, j])
                        T.sync_threads()
                        T.copy(tile[ib + 1, :, :], ks, prefer_instruction="cp_async")
                        T.sync_threads()
                        if gemm:
                            T.gemm(pf0, ks[:, 0:DH], of0)
                            T.gemm(pf1, ks[:, DH:2 * DH], of1)
                        else:
                            for i, j in T.Parallel(GQA, DH):
                                of1[i, j] += T.Cast("float32", ks[i, j])
                    else:
                        ib = T.max(T.min((bx * st + b) * 2, (1 << 21) - 2), 0)
                        T.copy(tile[ib, :, :], ks, prefer_instruction="cp_async")
                        T.copy(tile[ib + 1, :, :], vs, prefer_instruction="cp_async")
                    T.sync_threads()
                    if gemm and not single and not col:
                        T.clear(sf)
                        T.gemm(qs[0:GQA, :], ks[:, 0:DH], sf, transpose_B=True)
                        T.copy(sf, ps)
                        T.sync_threads()
                        T.gemm(ps, vs[:, 0:DH], of0)
                        T.clear(sf)
                        T.gemm(qs[GQA:2 * GQA, :], ks[:, DH:2 * DH], sf,
                               transpose_B=True)
                        T.copy(sf, ps)
                        T.sync_threads()
                        T.gemm(ps, vs[:, DH:2 * DH], of1)
                    elif not single and not col:
                        for i, j in T.Parallel(GQA, DH):
                            of0[i, j] += T.Cast("float32", ks[i, j])
                            of1[i, j] += T.Cast("float32", vs[i, j])
                T.copy(of0, fo[0:GQA, :])
                T.copy(of1, fo[GQA:2 * GQA, :])
                T.sync_threads()
                if bx == 0:
                    for i in T.serial(HQ):
                        if tid < DH:
                            o[i, tid] = fo[i, tid]
        return main
    return build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=16 * 128 // ABLK)
    a = ap.parse_args()
    torch.manual_seed(0)
    n = CTAS * a.nb * 2 * NEL
    buf = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    ptrs = torch.tensor([buf.data_ptr()] * 2, device="cuda", dtype=torch.int64)
    meta = torch.tensor([a.nb], device="cuda", dtype=torch.int32)
    q = torch.randn(HQ, DH, device="cuda", dtype=torch.float32)
    o = torch.zeros(HQ, DH, device="cuda", dtype=torch.float32)
    gb = n * 2 / 1e9
    floor = gb / 4.576
    print(f"{CTAS} CTAs x {a.nb} blocks   {gb:.3f} GB   "
          f"floor {floor * 1e3:.1f} us at 4.576 TB/s")
    for layout in ("swizzled", "single", "col"):
        for gemm in (False, True):
            fn = make(layout, gemm, a.nb)
            for _ in range(3):
                fn(ptrs, meta, q, o)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            N = 30
            for _ in range(N):
                fn(ptrs, meta, q, o)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / N
            print(f"  {layout:9s} {'gemm  ' if gemm else 'load  '}"
                  f"{dt * 1e6:8.1f} us   {gb / dt / 1e3:5.3f} TB/s"
                  f"   {dt / (a.nb * 6) * 1e9:7.0f} ns per block-layer", flush=True)
