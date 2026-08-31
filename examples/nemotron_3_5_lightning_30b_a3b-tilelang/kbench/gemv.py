"""How fast is one gemv pass of the mega kernel's shape, as a function of its size?

Every weight in the step goes through the same loop -- TMA a tile of `rb` rows
into shared, each thread walks a strided slice, reduce, store -- but the sizes
differ wildly: the head is 125 tiles per CTA, the router is 1. This runs `reps`
back-to-back passes over an (n, kd) matrix so the achieved bandwidth can be read
off against the size, which is what says whether the loop is throughput-bound or
paying full memory latency once per pass.
"""
import sys, time, importlib, pathlib, torch

import os
CTAS, THREADS = 132, 256
ARENA = int(os.environ.get('KB_ARENA', 90000))


def src(n, kd, reps, rb, nacc, red, xbf, vec, units=1, pad=0):
    lpr = THREADS // rb
    per, tile = kd // lpr, rb * kd
    it = per // vec
    nr = (n + CTAS - 1) // CTAS
    ntile = (nr + rb - 1) // rb
    ns = min(max(2, min(8, ARENA // tile)), ntile)
    xt = "bfloat16" if xbf else "float32"
    xc = (lambda s: f'T.Cast("float32", {s})') if xbf else (lambda s: s)
    tot = max(units, 1) * n * kd
    L, A = [], None
    L.append(f"""
import tilelang, tilelang.language as T
CTAS, THREADS = {CTAS}, {THREADS}
@tilelang.jit
def build():
    @T.prim_func
    def main(w: T.Tensor(({tot},), "bfloat16"), x: T.Tensor(({kd},), "{xt}"),
             sel: T.Tensor(({reps},), "int32"), out: T.Tensor(({n},), "float32")):
        with T.Kernel(CTAS, threads=THREADS) as cta:
            sm = T.alloc_shared(({ns * tile},), "bfloat16")
            fs = T.alloc_shared(({kd},), "{xt}")
            red = T.alloc_shared((THREADS,), "float32")
            pad_ = T.alloc_shared(({max(pad, 1)},), "float32")
            bar = T.alloc_barrier([1] * {ns})
            acc = T.alloc_local(({nacc},), "float32")
            tid = T.get_thread_binding()
            pad_[tid] = 0.0
            for _i in T.serial({(kd + THREADS - 1) // THREADS}):
                if _i * THREADS + tid < {kd}:
                    fs[_i * THREADS + tid] = x[_i * THREADS + tid]
            for _rep in T.serial({reps}):""")
    A = L.append
    P = "                "
    row = lambda t: f"T.max(T.min(cta * {nr} + ({t}) * {rb}, {n - rb}), 0)"
    off = lambda t: f'T.max(T.min(sel[_rep] * {n * kd} + {row(t)} * {kd}, {tot - tile}), 0)'
    for j in range(ns):
        A(P + "T.sync_threads()")
        A(P + f"T.tma_copy(w[{off(j)}:{off(j)} + {tile}],"
              f" sm[{j * tile}:{(j + 1) * tile}], barrier=bar[{j}])")
        A(P + f"if T.shuffle_elect(THREADS):\n{P}    T.barrier_arrive(bar[{j}])")
    A(P + "T.sync_threads()")
    A(P + f"for _t in T.serial({ntile}):")
    Q = P + "    "
    A(Q + f"T.mbarrier_wait_parity(bar[_t % {ns}], (_t // {ns}) % 2)")
    for a in range(nacc):
        A(Q + f"acc[{a}] = 0.0")
    A(Q + f"for _i in T.serial({it}):")
    for v in range(vec):
        e = f"(tid % {lpr}) * {vec} + _i * {lpr * vec} + {v}"
        A(Q + f"    acc[{v % nacc}] += {xc(f'fs[{e}]')} * T.Cast(\"float32\","
              f" sm[(_t % {ns}) * {tile} + (tid // {lpr}) * {kd} + {e}])")
    for a in range(1, nacc):
        A(Q + f"acc[0] += acc[{a}]")
    if red == "warp":
        A(Q + "acc[0] = T.warp_reduce_sum(acc[0])")
        if lpr > 32:
            A(Q + "red[tid] = acc[0]")
            A(Q + "T.sync_threads()")
            A(Q + f"if tid % {lpr} == 0:")
            A(Q + "    acc[0] = 0.0")
            A(Q + f"    for _q in T.serial({lpr // 32}):")
            A(Q + f"        acc[0] += red[tid + _q * 32]")
        else:
            A(Q + f"if tid % {lpr} == 0:")
        A(Q + f"    if {row('_t')} + tid // {lpr} >= cta * {nr}:")
        A(Q + f"        out[{row('_t')} + tid // {lpr}] = acc[0]")
    else:
        A(Q + "red[tid] = acc[0]")
        A(Q + "T.sync_threads()")
        A(Q + f"if tid < {rb}:")
        A(Q + "    acc[0] = 0.0")
        A(Q + f"    for _q in T.serial({lpr}):")
        A(Q + f"        acc[0] += red[tid * {lpr} + _q]")
        A(Q + f"    if {row('_t')} + tid >= cta * {nr}:")
        A(Q + f"        out[{row('_t')} + tid] = acc[0]")
    A(Q + "T.sync_threads()")
    A(Q + f"if _t + {ns} < {ntile}:")
    A(Q + f"    T.tma_copy(w[{off(f'_t + {ns}')}:{off(f'_t + {ns}')} + {tile}],"
          f" sm[(_t % {ns}) * {tile}:(_t % {ns}) * {tile} + {tile}], barrier=bar[_t % {ns}])")
    A(Q + "    if T.shuffle_elect(THREADS):")
    A(Q + f"        T.barrier_arrive(bar[_t % {ns}])")
    for s in range(ns):
        if ((ntile - s + ns - 1) // ns) % 2 == 1:
            A(P + f"if T.shuffle_elect(THREADS):\n{P}    T.barrier_arrive(bar[{s}])")
            A(P + f"T.mbarrier_wait_parity(bar[{s}], 1)")
    A("    return main")
    return "\n".join(L), ns, ntile, nr


def bench(n, kd, reps, rb=8, nacc=1, red="smem", xbf=False, vec=2, units=1, pad=0):
    code, ns, ntile, nr = src(n, kd, reps, rb, nacc, red, xbf, vec, units, pad)
    tag = f"_g{n}_{kd}_{reps}_{rb}_{nacc}_{red}_{int(xbf)}_{vec}_{units}_{pad}"
    pathlib.Path(f"kbench/{tag}.py").write_text(code)
    fn = importlib.import_module(f"kbench.{tag}").build()
    dt = torch.bfloat16 if xbf else torch.float32
    w = torch.randn(max(units, 1) * n * kd, device="cuda", dtype=torch.bfloat16)
    perm = (torch.randperm(units)[:reps] if units >= reps
            else torch.arange(reps) % max(units, 1))
    selt = perm.to("cuda", torch.int32)
    x = torch.randn(kd, device="cuda", dtype=dt)
    out = torch.zeros(n, device="cuda", dtype=torch.float32)
    for _ in range(3):
        fn(w, x, selt, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn(w, x, selt, out)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    ref = w.reshape(-1, kd)[int(perm[-1]) * n:int(perm[-1]) * n + 8].float() @ x.float()
    err = (out[:8] - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    read = reps * CTAS * ntile * rb * kd * 2          # what the CTAs actually pull
    print(f"n={n:6d} kd={kd:5d} tiles/cta={ntile:4d} stages={ns} rb={rb} nacc={nacc}"
          f" red={red:<4} x={'bf' if xbf else 'f3'} vec={vec} | {ms * 1e3 / reps:8.2f} us/pass"
          f"  {read / (ms * 1e-3) / 1e9:7.0f} GB/s  amp={read / (reps * n * kd * 2):.2f}"
          f"  rel {err:.1e}", flush=True)
    return ms / reps


if __name__ == "__main__":
    for a in eval(sys.argv[1]):
        bench(*a)
