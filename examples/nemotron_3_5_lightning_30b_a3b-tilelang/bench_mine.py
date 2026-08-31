"""Steady-state decode tok/s of the one-launch step, at every context length.

One sweep: the context is grown a token at a time and the clock is taken at each
checkpoint, because building a 262144-token context any other way would be a
different arithmetic from the one the table is about. Every point is warmed
before it is timed, and the SM clock is sampled while it is timed so the table
says what the card was doing, not what its spec sheet says.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch

import paths
#: The lengths SGLang was measured at, plus the two ends of the range.
POINTS = (0, 32, 1024, 4096, 16384, 32768, 65536, 131072, 262080)


def sm_clock(device=0):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
             "-i", str(device)], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default=None, help="or $NEMOTRON35_PREPARED")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--impl", default="mega", choices=["mega", "ops"])
    ap.add_argument("--steps", type=int, default=256, help="timed steps per point")
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--points", default=",".join(str(p) for p in POINTS))
    ap.add_argument("--out", default="reports/mine_baseline.json")
    a = ap.parse_args()
    os.environ["NEMO_IMPL"] = a.impl

    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415
    import runtime_model  # noqa: PLC0415

    points = sorted({int(p) for p in a.points.split(",")})
    top = max(points)
    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=a.device))
    cap = top + a.warmup + a.steps + 8
    caches = twin.init_caches(device=a.device, capacity=cap)
    ids = torch.full((cap,), 1784, dtype=torch.int64, device=a.device)

    rows, step, dev = [], 0, int(a.device.split(":")[-1]) if ":" in a.device else 0

    def advance(n):
        nonlocal step, caches
        for _ in range(n):
            args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches,
                                                      device=a.device)
            _, fresh = twin.forward(*args)
            caches = twin.append_cache(caches, fresh)
            step += 1

    for ctx in points:
        advance(ctx - step)
        advance(a.warmup)                      # warm this length before timing it
        torch.cuda.synchronize()
        clk0 = sm_clock(dev)
        t0 = time.perf_counter()
        advance(a.steps)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        clk1 = sm_clock(dev)
        rows.append({"ctx": ctx, "ms_per_token": dt * 1e3, "tok_s": 1.0 / dt,
                     "steps": a.steps, "sm_mhz": [clk0, clk1]})
        print(f"ctx {ctx:7d}  {1 / dt:8.2f} tok/s  {dt * 1e3:7.3f} ms/token"
              f"   sm {clk0}/{clk1} MHz", flush=True)

    json.dump({"impl": a.impl, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
