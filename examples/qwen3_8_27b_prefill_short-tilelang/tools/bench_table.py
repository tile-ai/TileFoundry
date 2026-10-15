#!/usr/bin/env python3
"""BENCH.md's table from one same-session alternating run (bench/alternate.py output)
and the FLOP ledger (reports/flops.json).

usage: python tools/bench_table.py bench/results/<stamp>.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PEAK = 989.5e12


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    ledger = {r["T"]: r for r in json.loads((ROOT / "reports/flops.json").read_text())["rows"]}
    gpu = run["load"]["sglang"]["gpu"]
    card = f"GPU {run['gpu_physical_index']} ({gpu['uuid'][:12]}..., bus {gpu['pci_bus_id'][-7:]})"
    print(f"Run `{Path(sys.argv[1]).name}`: rounds {run['args']['rounds']} x iters {run['args']['iters']} per side "
          f"(warm-up {run['args']['warmup']} per call), card {card}, both workers started by one driver in one session.\n")
    print("| T | mine ms median [min..max] | SGLang ms median [min..max] | SGLang / mine | roofline lower bound ms "
          "| mine TFLOP/s | % of 989.5 | SM clock MHz during timed iters (mine; SGLang) | card | first token mine = SGLang = HF |")
    print("|---:|---|---|---:|---:|---:|---:|---|---|---|")
    for T, row in sorted(run["rows"].items(), key=lambda kv: int(kv[0])):
        T = int(T)
        m, s = row["mine"], row["sglang"]
        led = ledger[T]
        tf = led["total_flops"] / (m["median_ms"] * 1e-3) / 1e12
        toks = {tuple(set(m["first_tokens"])), tuple(set(s["first_tokens"]))}
        same = len(toks) == 1 and list(toks)[0] == (row["reference_first_token"],)
        print(f"| {T} | {m['median_ms']:.2f} [{m['min_ms']:.2f}..{m['max_ms']:.2f}] "
              f"| {s['median_ms']:.2f} [{s['min_ms']:.2f}..{s['max_ms']:.2f}] | {s['median_ms'] / m['median_ms']:.2f}x "
              f"| {led['roofline_ms']:.2f} ({led['bound']}) | {tf:.0f} | {100 * tf * 1e12 / PEAK:.1f} % "
              f"| {m['sm_clock_min']}..{m['sm_clock_max']}; {s['sm_clock_min']}..{s['sm_clock_max']} "
              f"| GPU {run['gpu_physical_index']} | {row['reference_first_token']} {'yes' if same else 'NO'} |")


if __name__ == "__main__":
    main()
