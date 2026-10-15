"""Step-3 evidence: fusion alternatives of the GDN mixer (between the in-projection GEMM
output and the out-projection input), GPU time per T.

  (A) prep:  fused gdn_prep (conv + silu + l2norm + v + g + beta + window)
             vs split: small.conv_silu_kernel + small.l2norm_kernel(q) + small.l2norm_kernel(k)
                       + gb_window_kernel (g, beta, conv window; defined here)
  (B) norm:  gdn_scan with the fused gated RMSNorm (GdnKernels fused_norm=True path)
             vs plain gdn_scan + gdn_gated_norm (the production path of gdn.py since the
             fusion study; the kernel first lived here)
  (C) chunk + scan in one kernel: no such variant exists; not measured.

    CUDA_VISIBLE_DEVICES=1 python runtime/experiments/gdn_fusion.py [--T 128,256,512,1024]
        [--iters 50] [--no-wait]      (--iters 0: checks only, no timing, no JSON)

Timing "graph": the method of runtime/kernels/test_gdn.py -- the piece is captured alone
in a CUDA graph, a GPU sleep is queued, events around one replay, median of `iters`
replays after warm-up (includes ~3-4 us of graph-launch latency per graph).
"chain" = all kernels of an alternative in one graph.  "b2b" = runtime.benchutil
.graph_time_us (20 back-to-back calls in one graph, per-call time, no launch latency).
Before each T the script waits (60 s polls) until GPU1 has no other compute process.

Inputs: realistic layer-0 inputs of test_gdn (proj buffer of row stride 16640).
small.conv_silu_kernel and small.l2norm_kernel take contiguous tensors only, so the split
path gets contiguous copies of `mixed` and of q/k (made outside the timed region); the
kernels read the same number of bytes as on strided views.  Results are checked against
the fused kernels (prep: bitwise; norm: element count that differs).
Results: reports/gdn_fusion.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

from runtime.benchutil import graph_time_us  # noqa: E402
from runtime.kernels import small  # noqa: E402
from runtime.kernels.gdn import GdnKernels  # noqa: E402
from runtime.kernels.test_gdn import (PAD_STRIDE, build_inputs, gate, load_layer_weights,  # noqa: E402
                                      sm_clock, time_graph)

BF16, F32 = "bfloat16", "float32"
HV, D, CONV, PROJ = 48, 128, 10240, 16480
Z_OFF, B_OFF, A_OFF = 10240, 16384, 16432
GPU1_UUID = "GPU-3a3112ff-837c-3b87-38d1-cb203d656589"


def gb_window_kernel(TT, PS=PAD_STRIDE, threads=256):
    """g = -exp(a_log) * softplus(a + dt_bias) (f32), beta = bf16(sigmoid(b)), window[c, j] =
    mixed[T-3+j, c]; the same expressions as gdn_prep (bitwise equal to the HIR)."""
    NGB = TT * HV // threads
    NW = CONV * 3 // threads
    assert NGB * threads == TT * HV and NW * threads == CONV * 3

    @T.prim_func
    def gb_window(projb: T.StridedTensor((TT, PROJ), (PS, 1), BF16), a_log: T.Tensor((HV,), BF16),
                  dt_bias: T.Tensor((HV,), BF16), g: T.Tensor((TT, HV), F32), beta: T.Tensor((TT, HV), BF16),
                  window: T.Tensor((CONV, 3), BF16)):
        with T.Kernel(NGB + NW, threads=threads) as bx:
            tx = T.get_thread_binding()
            if bx < NGB:
                e = bx * threads + tx
                t = e // HV
                h = e % HV
                av = T.cast(projb[t, A_OFF + h], F32) + T.cast(dt_bias[h], F32)
                sp = T.if_then_else(av > 20.0, av, T.log1p(T.exp(av)))
                g[t, h] = -T.exp(T.cast(a_log[h], F32)) * sp
                bv = T.cast(projb[t, B_OFF + h], F32)
                beta[t, h] = T.cast(1.0 / (1.0 + T.exp(-bv)), BF16)
            else:
                e = (bx - NGB) * threads + tx
                c = e // 3
                j = e % 3
                window[c, j] = projb[TT - 3 + j, c]

    return tilelang.compile(gb_window, target="cuda")


def wait_gpu1_free(poll_s=60):
    """Wait until GPU1 has no compute process other than this one."""
    me = os.getpid()
    while True:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"],
                              capture_output=True, text=True).stdout
        util = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        others = [ln for ln in apps.splitlines() if ln.startswith(GPU1_UUID) and int(ln.split(",")[1]) != me]
        if not others:
            print(f"   GPU1 free (utilization {util})", flush=True)
            return
        print(f"   GPU1 busy ({len(others)} other process(es), utilization {util}); waiting {poll_s} s", flush=True)
        time.sleep(poll_s)


def bitwise_diff(a, b):
    a = a.contiguous().view(-1)
    b = b.contiguous().view(-1)
    it = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return int((a.view(it) != b.view(it)).sum().item())


def run(TT, W, iters, wait):
    dev = "cuda"
    buf = build_inputs(TT, 0, W)
    proj = buf[:, :PROJ]
    z3 = buf[:, Z_OFF:Z_OFF + HV * D].view(TT, HV, D)
    conv_w = W["conv_w"].view(CONV, 4)
    kern = GdnKernels(TT, proj_stride=PAD_STRIDE)

    def bf(*s):
        return torch.empty(*s, dtype=torch.bfloat16, device=dev)

    # fused prep outputs
    q, k, v = bf(TT, 16, D), bf(TT, 16, D), bf(TT, HV, D)
    g = torch.empty(TT, HV, device=dev)
    beta, window = bf(TT, HV), bf(CONV, 3)
    fused_prep = lambda: kern.prep(proj, conv_w, W["a_log"], W["dt_bias"], q, k, v, g, beta, window)  # noqa: E731

    # split prep
    k_conv = small.conv_silu_kernel(TT, CONV)
    k_l2 = small.l2norm_kernel(TT, 16, D)
    k_gbw = gb_window_kernel(TT)
    mixed_c = buf[:, :CONV].contiguous()
    conv_out = bf(TT, CONV)
    q2, k2 = bf(TT, 16, D), bf(TT, 16, D)
    g2 = torch.empty(TT, HV, device=dev)
    beta2, window2 = bf(TT, HV), bf(CONV, 3)
    s_conv = lambda: k_conv(mixed_c, conv_w, conv_out)  # noqa: E731
    fused_prep()
    s_conv()
    torch.cuda.synchronize()
    q_in = conv_out[:, :2048].reshape(TT, 16, D).contiguous()
    k_in = conv_out[:, 2048:4096].reshape(TT, 16, D).contiguous()
    s_l2q = lambda: k_l2(q_in, q2)  # noqa: E731
    s_l2k = lambda: k_l2(k_in, k2)  # noqa: E731
    s_gbw = lambda: k_gbw(proj, W["a_log"], W["dt_bias"], g2, beta2, window2)  # noqa: E731

    def split_prep():
        s_conv(); s_l2q(); s_l2k(); s_gbw()  # noqa: E702

    split_prep()
    torch.cuda.synchronize()
    chk_a = {"q": bitwise_diff(q2, q), "k": bitwise_diff(k2, k),
             "v": bitwise_diff(conv_out[:, 4096:].reshape(TT, HV, D), v),
             "g": bitwise_diff(g2, g), "beta": bitwise_diff(beta2, beta), "window": bitwise_diff(window2, window)}

    # scan: fused gated norm vs plain scan + gated_norm
    k_gn = kern.gated_norm_kernel()
    y_f, y_s, o = bf(TT, HV, D), bf(TT, HV, D), bf(TT, HV, D)
    st_f = torch.empty(HV, D, D, device=dev)
    st_p = torch.empty(HV, D, D, device=dev)
    kern.core(q, k, v, g, beta, z3, W["gamma_gdn"], y_f, st_f)
    ws = (kern.ws_ut, kern.ws_w, kern.ws_p, kern.ws_f)
    scan_plain = kern.scan_plain_kernel()
    chunk = lambda: kern.chunk_kernel(q, k, v, g, beta, *ws)  # noqa: E731
    scan_fused = lambda: kern.fused_scan_kernel()(q, k, *ws, z3, W["gamma_gdn"], y_f, st_f)  # noqa: E731
    scan_p = lambda: scan_plain(q, k, *ws, o, st_p)  # noqa: E731
    norm = lambda: k_gn(o, z3, W["gamma_gdn"], y_s)  # noqa: E731
    chunk(); scan_fused(); scan_p(); norm()  # noqa: E702
    torch.cuda.synchronize()
    y_hir = gate(o, buf[:, Z_OFF:Z_OFF + HV * D], W["gamma_gdn"], TT).view(TT, HV, D)
    chk_b = {"split_vs_hir_formula_differing": bitwise_diff(y_s, y_hir),
             "fused_vs_hir_formula_differing": bitwise_diff(y_f, y_hir),
             "fused_vs_split_maxabs": (y_f.float() - y_s.float()).abs().max().item(),
             "elements": TT * HV * D}

    if iters == 0:
        return {"check_prep_split_vs_fused_differing": chk_a, "check_norm": chk_b}
    if wait:
        wait_gpu1_free()
    clk = sm_clock()

    def tg(fn):
        return round(time_graph(fn, iters)[0], 2)

    def tb(fn):
        return round(graph_time_us(fn), 2)

    pieces = {
        "prep_fused": fused_prep, "conv_silu": s_conv, "l2norm_q": s_l2q, "l2norm_k": s_l2k,
        "g_beta_window": s_gbw, "prep_split_chain": split_prep,
        "chunk": chunk, "scan_fused_norm": scan_fused, "scan_plain": scan_p, "gated_norm": norm,
        "scan_plain_plus_norm_chain": lambda: (scan_p(), norm()),
        "core_fused": lambda: (chunk(), scan_fused()),
        "core_split_chain": lambda: (chunk(), scan_p(), norm()),
    }
    graph = {n: tg(f) for n, f in pieces.items()}
    b2b = {n: tb(f) for n, f in pieces.items()}
    for d in (graph, b2b):
        d["prep_split_sum"] = round(d["conv_silu"] + d["l2norm_q"] + d["l2norm_k"] + d["g_beta_window"], 2)
        d["scan_plain_plus_norm_sum"] = round(d["scan_plain"] + d["gated_norm"], 2)
    return {"graph_us": graph, "b2b_us": b2b, "check_prep_split_vs_fused_differing": chk_a,
            "check_norm": chk_b, "sm_clock": clk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="128,256,512,1024")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "gdn_fusion.json"))
    args = ap.parse_args()
    W = load_layer_weights(0)
    res = {"method": {
        "graph": "one CUDA graph per piece, GPU sleep before, events around one replay, median of "
                 f"{args.iters} replays (as test_gdn.py; includes ~3-4 us graph launch per graph)",
        "b2b": "runtime.benchutil.graph_time_us: 20 back-to-back calls per graph, per-call time",
        "chain": "all kernels of the split alternative in one graph",
        "inputs": "layer 0 realistic inputs, proj row stride 16640; split prep kernels get contiguous "
                  "copies of mixed and q/k (small.* kernels need contiguous tensors)",
        "C_chunk_scan_one_kernel": "not measured: no fused chunk+scan variant exists",
    }}
    for TT in [int(x) for x in args.T.split(",")]:
        print(f"=== T={TT}", flush=True)
        r = run(TT, W, args.iters, not args.no_wait)
        res[f"T{TT}"] = r
        print("   checks", r["check_prep_split_vs_fused_differing"], r["check_norm"], flush=True)
        for n in r.get("graph_us", {}):
            print(f"   {n:28s} graph {r['graph_us'][n]:8.2f} us   b2b {r['b2b_us'][n]:8.2f} us", flush=True)
        torch.cuda.empty_cache()
    if args.iters == 0:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
