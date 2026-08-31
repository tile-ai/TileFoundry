"""Two questions the integration turns on.

1. Does the shared allocator reuse space between buffers whose lifetimes are
   disjoint? The gemv arena is 180 KB and the attention operands another 144;
   both at once is past the 227 KB a CTA has, but they are never live together.
2. Can a strided slice of one shared buffer be a WGMMA operand? The bulk copy
   lands K and V as (128 positions, 2 KV heads x 128 dims); a group is a column
   slice of that, and copying it out again would cost a shared round trip.
"""
import sys
import torch
import tilelang
import tilelang.language as T

POS, DH, HEADS = 128, 128, 16
WHICH = sys.argv[1] if len(sys.argv) > 1 else "reuse"


@tilelang.jit
def build_reuse():
    @T.prim_func
    def main(x: T.Tensor((1024,), "bfloat16"), out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=256) as bx:
            arena = T.alloc_shared((90000,), "bfloat16")      # 180 KB
            ks = T.alloc_shared((POS, 2 * DH), "bfloat16")    #  64 KB
            vs = T.alloc_shared((POS, 2 * DH), "bfloat16")    #  64 KB
            tid = T.get_thread_binding()
            arena[tid] = x[tid % 1024]
            T.sync_threads()
            ks[tid // 256, tid % 256] = arena[tid]
            vs[tid // 256, tid % 256] = arena[tid]
            T.sync_threads()
            if tid < 8:
                out[tid] = T.Cast("float32", ks[0, tid] + vs[0, tid])
    return main


@tilelang.jit
def build_slice():
    @T.prim_func
    def main(q: T.Tensor((HEADS, DH), "bfloat16"), kv: T.Tensor((POS, 2 * DH), "bfloat16"),
             s_out: T.Tensor((HEADS, POS), "float32")):
        with T.Kernel(1, threads=256) as bx:
            qs = T.alloc_shared((HEADS, DH), "bfloat16")
            ks = T.alloc_shared((POS, 2 * DH), "bfloat16")
            sf = T.alloc_fragment((HEADS, POS), "float32")
            T.copy(q, qs)
            T.copy(kv, ks)
            T.clear(sf)
            T.gemm(qs, ks[:, DH:2 * DH], sf, transpose_B=True)
            T.copy(sf, s_out)
    return main


if __name__ == "__main__":
    if WHICH == "reuse":
        try:
            build_reuse()
            print("REUSE OK: 180 KB arena + 128 KB of operands compiled together")
        except Exception as e:
            print("REUSE FAIL:", str(e).splitlines()[-1][:140])
    else:
        torch.manual_seed(0)
        q = torch.randn(HEADS, DH, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(POS, 2 * DH, device="cuda", dtype=torch.bfloat16)
        s = torch.zeros(HEADS, POS, device="cuda", dtype=torch.float32)
        try:
            build_slice()(q, kv, s)
            torch.cuda.synchronize()
            ref = q.float() @ kv[:, DH:].float().t()
            print(f"SLICE OK: rel_l2 {((s - ref).norm() / ref.norm()).item():.3e}")
        except Exception as e:
            print("SLICE FAIL:", str(e).splitlines()[-1][:140])
