"""Does the shared allocator reuse space across stages that alternate in a loop?

The mega kernel's layers alternate: a streaming stage fills the arena, an
attention stage fills its own operands, and round again.  Straight-line code
makes the two lifetimes obviously disjoint; a loop does not, and whether the
allocator still overlaps them is the difference between 128 KB of tensor-core
operands being free and being impossible.
"""
import sys
import torch
import tilelang
import tilelang.language as T

POS, DH, KVROW = 128, 128, 256
ARENA = int(sys.argv[1]) if len(sys.argv) > 1 else 90112


@tilelang.jit
def build():
    @T.prim_func
    def main(x: T.Tensor((1024,), "bfloat16"), out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=256) as bx:
            arena = T.alloc_shared((ARENA,), "bfloat16")
            ks = T.alloc_shared((POS, KVROW), "bfloat16")
            vs = T.alloc_shared((POS, KVROW), "bfloat16")
            tid = T.get_thread_binding()
            for _l in T.serial(4):
                for r in T.serial(ARENA // 256):
                    arena[r * 256 + tid] = x[(tid + r + _l) % 1024]
                T.sync_threads()
                for r in T.serial(POS * KVROW // 256):
                    ks[(r * 256 + tid) // KVROW, (r * 256 + tid) % KVROW] = arena[r * 256 + tid]
                    vs[(r * 256 + tid) // KVROW, (r * 256 + tid) % KVROW] = arena[r * 256 + tid + 1]
                T.sync_threads()
                if tid < 8:
                    out[tid] = T.Cast("float32", ks[0, tid]) + T.Cast("float32", vs[1, tid])
                T.sync_threads()
    return main


if __name__ == "__main__":
    need = (ARENA + 2 * POS * KVROW) * 2 / 1024
    try:
        fn = build()
        x = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
        o = torch.zeros(8, device="cuda", dtype=torch.float32)
        fn(x, o)
        torch.cuda.synchronize()
        print(f"REUSE OK   arena {ARENA * 2 / 1024:.0f} KB + operands 128 KB"
              f" = {need:.0f} KB declared, fits in 227")
    except Exception as e:
        print(f"REUSE FAIL {need:.0f} KB declared: " + str(e).strip().splitlines()[-1][:160])
