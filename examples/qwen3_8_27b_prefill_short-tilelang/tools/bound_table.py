#!/usr/bin/env python3
"""Bound type of every kernel of one forward: measured GPU time (runtime/profile_forward.py
output) next to its compute floor and its memory floor.

  compute floor = FLOP / PEAK_TC   (bf16 tensor cores at the SM clock actually used:
                  132 SM x 4096 FLOP/clk x 1.5 GHz = 811 TFLOP/s; 989.5 is the 1.83 GHz value)
  memory floor  = bytes the kernel must move through HBM / 4.8 TB/s
                  (weights once, activations in and out once, f32 partials and
                  workspaces written and read; L2 re-reads are not counted)
  bound         = the larger floor; efficiency = floor / measured; gap = measured - floor

Shapes come from the kernel names (gemm_<epi>_<M>x<N>x<K>_<cfg>, ...). GDN and attention
bytes follow the buffers in runtime/kernels/{gdn,attention}.py; their FLOP from
reports/flops.json (the tensor-core part only).

usage: python tools/bound_table.py [T ...]  (reads reports/profile/final_T<T>.txt)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PEAK_TC = 132 * 4096 * 1.5e9
HBM = 4.8e12
H, V = 5120, 248320


def model(name, T, ledger):
    """(FLOP, bytes) of one call of kernel `name` at prompt length T."""
    m = re.match(r"gemm_(\w+?)_(\d+)x(\d+)x(\d+)_.*?(?:_sk(\d+))?_kernel$", name)
    if m:
        epi, M, N, K = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        S = int(m.group(5) or 1)
        if epi == "swiglu2":                       # N is one half; two weights
            return 2 * M * 2 * N * K, 2 * N * K * 2 + M * K * 2 + M * N * 2
        b = N * K * 2 + M * K * 2
        if epi == "splitk":
            return 2 * M * N * K, b + S * M * N * 4
        if epi == "residual":
            return 2 * M * N * K, b + 2 * M * N * 2
        return 2 * M * N * K, b + M * N * 2
    m = re.match(r"splitk_reduce_(\d+)x(\d+)_s(\d+)(_res)?_kernel$", name)
    if m:
        M, N, S = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return 0, S * M * N * 4 + M * N * 2 * (2 if m.group(4) else 1)
    m = re.match(r"ba_splitk_(\d+)x(\d+)x(\d+)_s(\d+)_kernel$", name)
    if m:
        M, N, K, S = (int(m.group(i)) for i in range(1, 5))
        return 2 * M * N * K, N * K * 2 + M * K * 2 + S * M * N * 4
    m = re.match(r"ba_reduce_(\d+)_s(\d+)_kernel$", name)
    if m:
        M, S = int(m.group(1)), int(m.group(2))
        return 0, S * M * 96 * 4 + M * 96 * 2
    m = re.match(r"gemv_(\d+)x(\d+)_kernel$", name)
    if m:
        N, K = int(m.group(1)), int(m.group(2))
        return 2 * N * K, N * K * 2
    m = re.match(r"rmsnorm_(\d+)x(\d+)_kernel$", name)
    if m:
        M, K = int(m.group(1)), int(m.group(2))
        return 0, 2 * M * K * 2 + K * 2
    gdn_core = ledger["flops"]["gdn core (chunked WY, C=64)"] / 48
    att_core = ledger["flops"]["attention core (causal QK^T + PV)"] / 16
    if name == "gdn_prep_kernel":                  # qkv|b|a in, q k v g beta out, window
        return 0, T * (10336 * 2 + (2048 + 2048 + 6144) * 2 + 48 * 6) + 10240 * 3 * 2
    if name == "gdn_chunk_kernel":                 # k v q g beta in; U^T f32, W hi+lo, P, F out
        return gdn_core / 2, T * ((2048 * 2 + 6144) * 2 + 48 * 6 + 6144 * 4 + 6144 * 4 + 48 * 64 * 2)
    if name == "gdn_scan_plain_kernel":            # U W P F q k in; read-out out; state out
        return gdn_core / 2, T * (6144 * 4 + 6144 * 4 + 48 * 64 * 2 + 2048 * 2 * 2 + 6144 * 2) + 48 * 128 * 128 * 4
    if name == "gdn_gated_norm_kernel":            # read-out, z in; y out
        return 0, T * 6144 * 2 * 3
    if name == "attn_prep_kernel":                 # q (no gate) k v in and out
        return 0, 2 * T * (6144 + 1024 + 1024) * 2
    if name == "attn_core_kernel":                 # q gate in, k v once, y out
        return att_core, T * (6144 * 3 + 2048) * 2
    if name.startswith("embed_"):
        return 0, 2 * T * H * 2
    return None


def table(T):
    ledger = {r["T"]: r for r in json.loads((ROOT / "reports/flops.json").read_text())["rows"]}[T]
    rows, other = [], 0.0
    for line in (ROOT / f"reports/profile/final_T{T}.txt").read_text().splitlines():
        m = re.match(r"\s+([\d.]+) ms\s+[\d.]+%\s+x\s+(\d+)\s+(\S+)", line)
        if not m:
            continue
        ms, n, name = float(m.group(1)), int(m.group(2)), m.group(3)
        fb = model(name, T, ledger)
        if fb is None:
            other += ms
            continue
        f, b = fb
        c_ms, m_ms = n * f / PEAK_TC * 1e3, n * b / HBM * 1e3
        floor = max(c_ms, m_ms)
        rows.append(dict(name=name, calls=n, ms=ms, compute_ms=c_ms, memory_ms=m_ms, floor_ms=floor,
                         bound="compute" if c_ms >= m_ms else "memory", eff=floor / ms, gap_ms=ms - floor,
                         TFs=n * f / (ms * 1e-3) / 1e12, TBs=n * b / (ms * 1e-3) / 1e12))
    return rows, other


def main():
    lengths = [int(a) for a in sys.argv[1:]] or [128, 256, 512, 1024]
    out = {}
    for T in lengths:
        rows, other = table(T)
        out[T] = rows
        tot = sum(r["ms"] for r in rows) + other
        flo = sum(r["floor_ms"] for r in rows)
        print(f"\n### T = {T}: measured {tot:.2f} ms, sum of per-kernel floors {flo:.2f} ms "
              f"(efficiency {flo / tot:.0%}), gap {tot - flo:.2f} ms\n")
        print("| kernel | calls | measured ms | compute floor ms | memory floor ms | bound | efficiency | gap ms | TFLOP/s | TB/s |")
        print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda r: -r["gap_ms"]):
            print(f"| `{r['name'].removesuffix('_kernel')}` | {r['calls']} | {r['ms']:.3f} | {r['compute_ms']:.3f} | "
                  f"{r['memory_ms']:.3f} | {r['bound']} | {r['eff']:.0%} | {r['gap_ms']:.3f} | {r['TFs']:.0f} | {r['TBs']:.2f} |")
    (ROOT / "reports/bound_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
