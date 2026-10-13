"""Prefill latency of the tilelang runtime twin, by bench_prefill.py's protocol.

    python tools/bench.py [--lens 1024,4096,8192] [--out reports/bench.json]

What is timed: one CUDA graph replay of the whole prefill -- token ids
already on the device to the first token on the device (embedding, 64
layers, closing norm, lm_head row, argmax) -- bracketed by
`torch.cuda.synchronize()`. Weight loading, kernel compilation, graph
capture and tokenization happen before and are not timed. This is the
"forward only" scope: no scheduler, no detokenize, no IPC.

Repeats, as bench_prefill.py: 2 untimed warmup replays per length, then 7
timed, median reported with min / max; the SM clock is read right after each
timed replay, while the GPU is still hot, and recorded, not used to correct
anything. Each repeat uses a different prompt token sequence (the prompt
rotated by one), so no two repeats see the same ids.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import flops as flopmod  # noqa: E402
import run as RUN  # noqa: E402
from prompt import CKPT, DEFAULT_TEXT, prompt_ids  # noqa: E402

PEAK = 989.0          # NVIDIA's BF16 dense figure
PEAK_MHZ_1980 = 1980.0  # the clock SGLANG_BASELINE.md divides by
FLOP_PER_CLK = 132 * 4096  # measured: 4096 dense bf16 FLOP/clk/SM (memory h200-power-clock)


def sm_clock_mhz() -> float:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def others_on_card() -> list[str]:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "-i", dev, "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout.split()
    return [p for p in out if p != str(os.getpid())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1024,4096,8192")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "reports" / "bench.json"))
    ap.add_argument("--label", default="tilelang-twin")
    ap.add_argument("--gemm-split", choices=("rule", "on", "off"), default="rule",
                    help="segmentation study: force the GEMM split tail on / off")
    ap.add_argument("--gemm-group", type=int, default=0,
                    help="segmentation study: force the raster GROUP (where it divides the M-tiles)")
    args = ap.parse_args()
    from kernels import gemm as G
    G.OVERRIDE["split_tail"] = {"rule": None, "on": True, "off": False}[args.gemm_split]
    G.OVERRIDE["group"] = args.gemm_group or None

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    twin = RUN.load_twin(CKPT)
    rows = []
    for n in [int(x) for x in args.lens.split(",")]:
        base = prompt_ids(tok, DEFAULT_TEXT, n + args.warm + args.reps).cuda()
        pre = RUN.Prefill(twin, n)
        variant = lambda k: base[k:k + n]  # noqa: E731
        for k in range(args.warm):
            pre(variant(k), n - 1)
        torch.cuda.synchronize()
        lat, clk, firsts, others = [], [], [], []
        for k in range(args.reps):
            ids = variant(args.warm + k)
            pre.ids.copy_(ids)
            pre.last.fill_(n - 1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            pre.graph.replay()
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t)
            clk.append(sm_clock_mhz())
            firsts.append(int(pre.token))
            others.extend(others_on_card())
        med = statistics.median(lat)
        f = flopmod.prefill_flops(Path(CKPT), n)
        tflops = f["total"] / med / 1e12
        mclk = statistics.median(clk)
        row = {
            "n": n, "latency_ms": med * 1e3, "min_ms": min(lat) * 1e3, "max_ms": max(lat) * 1e3,
            "spread_ms": (max(lat) - min(lat)) * 1e3, "all_ms": [x * 1e3 for x in lat],
            "prefill_tok_s": n / med, "flop": f["total"], "tflop_s": tflops,
            "pct_989": 100 * tflops / PEAK,
            "pct_at_clock_1980": 100 * tflops / (PEAK * mclk / PEAK_MHZ_1980),
            "pct_at_clock_4096": 100 * tflops / (FLOP_PER_CLK * mclk * 1e6 / 1e12),
            "floor_ms": f["total"] / (PEAK * 1e12) * 1e3,
            "sm_clock_mhz": mclk, "sm_clocks": clk, "reps": args.reps, "warm": args.warm,
            "first_tokens": firsts, "other_processes_on_card": sorted(set(others)),
        }
        rows.append(row)
        print(f"n={n:>5} {row['latency_ms']:8.2f} ms (min {row['min_ms']:.2f} max {row['max_ms']:.2f} spread {row['spread_ms']:.2f}) "
              f"{row['prefill_tok_s']:7.0f} tok/s {tflops:6.1f} TFLOP/s {row['pct_989']:5.1f}% of 989 "
              f"floor {row['floor_ms']:.1f} ms sm {mclk:.0f} MHz clocks {clk} others {row['other_processes_on_card']}",
              flush=True)
        del pre
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"label": args.label, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gemm_override": dict(G.OVERRIDE),
               "when": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
