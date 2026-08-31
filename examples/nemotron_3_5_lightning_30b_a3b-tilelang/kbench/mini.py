"""Smallest TMA + mbarrier pipeline that must be right before anything else is."""
import sys, torch, tilelang, tilelang.language as T

CTAS = 4
TILE = 2048          # bf16 elements = 4 KB
STAGES = 2
THREADS = 128
NT = 8               # tiles per CTA
PER = TILE * NT
TOTAL = CTAS * PER


@tilelang.jit(out_idx=[-1])
def build():
    @T.prim_func
    def main(W: T.Tensor((TOTAL,), "bfloat16"), acc: T.Tensor((CTAS,), "float32")):
        with T.Kernel(CTAS, threads=THREADS) as bx:
            smem = T.alloc_shared((STAGES * TILE,), "bfloat16")
            bar = T.alloc_barrier([1] * STAGES)
            s = T.alloc_local((1,), "float32")
            tid = T.get_thread_binding()
            s[0] = 0.0
            base = bx * PER
            for j in T.serial(STAGES):
                T.tma_copy(W[base + j * TILE:base + (j + 1) * TILE],
                           smem[j * TILE:(j + 1) * TILE], barrier=bar[j])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[j])
            for i in T.serial(NT):
                st = i % STAGES
                T.mbarrier_wait_parity(bar[st], (i // STAGES) % 2)
                for r in T.serial(TILE // THREADS):
                    s[0] += T.Cast("float32", smem[st * TILE + r * THREADS + tid])
                T.sync_threads()
                if i + STAGES < NT:
                    g = base + (i + STAGES) * TILE
                    T.tma_copy(W[g:g + TILE], smem[st * TILE:(st + 1) * TILE], barrier=bar[st])
                    if T.shuffle_elect(THREADS):
                        T.barrier_arrive(bar[st])
            acc[bx] = s[0]
    return main


if __name__ == "__main__":
    fn = build()
    if "--src" in sys.argv:
        print(fn.get_kernel_source())
        raise SystemExit(0)
    W = torch.arange(TOTAL, device="cuda", dtype=torch.float32).reshape(-1) % 7
    W = W.to(torch.bfloat16)
    out = fn(W)
    torch.cuda.synchronize()
    ref = W.float().view(CTAS, PER).sum(1)
    print("got", out[:4].tolist())
    print("ref", ref.tolist())
