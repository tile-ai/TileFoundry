#!/usr/bin/env python3
"""STEP 3 segmentation evidence, whole forward: serve every prompt length T with the
GEMM plan of every planned length P (one shared set = the same P on every row; the
per-length set = the diagonal), and serve T by the engine of 2T (pad to the next
length = one implementation fewer).

Time = the runtime twin's prefill (CUDA-graph replay of the whole forward), CUDA
events, median of N after W warm-ups, as bench/mine_worker.py. First token recorded
so a plan swap that changed results would show.

usage: CUDA_VISIBLE_DEVICES=0 python runtime/experiments/segment_xplan.py [--iters 20]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import prompting  # noqa: E402
from runtime import engine as eng  # noqa: E402
from runtime import plan  # noqa: E402

LENGTHS = (128, 256, 512, 1024)


_PHYS = None


def physical_index():
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        idx, u = [x.strip() for x in line.split(",")]
        if u.removeprefix("GPU-") == uuid.removeprefix("GPU-"):
            return int(idx)


def sm_clock():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i",
                               str(_PHYS)], capture_output=True, text=True, check=True).stdout.strip())


CLOCKS = []


def timed(twin, ids, cos, sin, warmup, iters):
    for _ in range(warmup):
        out = twin(ids, cos, sin)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        out = twin(ids, cos, sin)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
        CLOCKS.append(sm_clock())
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1], int(torch.argmax(out[0].float()))


def ref_ids(T):
    return torch.load(ROOT / f"reference/T{T}/token_ids.pt").to("cuda").long().flatten().contiguous()


def drop_engines():
    eng._ENGINES.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    from runtime.twin import Qwen3_8_27BTwin
    from tilefoundry.runtime import SafetensorsResource

    global _PHYS
    torch.cuda.set_device(0)
    _PHYS = physical_index()
    twin = Qwen3_8_27BTwin()
    twin.load(SafetensorsResource(str(ROOT / "ckpt"), device="cuda"))
    real_for_length = plan.for_length
    results = {"cross_plan": [], "pad_to_next": []}
    for T in LENGTHS:
        ids = ref_ids(T)
        cos, sin = prompting.rope_tables(T, eng.CFG, "cuda")
        for P in LENGTHS:
            plan.for_length = lambda t, P=P: real_for_length(P)
            drop_engines()
            t0 = time.perf_counter()
            CLOCKS.clear()
            med, lo, hi, tok = timed(twin, ids, cos, sin, args.warmup, args.iters)
            row = dict(T=T, plan=P, median_ms=med, min_ms=lo, max_ms=hi, first_token=tok,
                       setup_s=time.perf_counter() - t0, sm_clock_min=min(CLOCKS), sm_clock_max=max(CLOCKS))
            results["cross_plan"].append(row)
            print(json.dumps(row), flush=True)
    plan.for_length = real_for_length
    for T in LENGTHS[:-1]:
        # T real tokens served by the 2T engine: the forward is that of 2T (the first
        # token would be read at row T-1; the GEMV over one row costs the same).
        drop_engines()
        T2 = 2 * T
        ids = ref_ids(T2)
        cos, sin = prompting.rope_tables(T2, eng.CFG, "cuda")
        CLOCKS.clear()
        med, lo, hi, tok = timed(twin, ids, cos, sin, args.warmup, args.iters)
        row = dict(T=T, served_by=T2, median_ms=med, min_ms=lo, max_ms=hi, sm_clock_min=min(CLOCKS),
                   sm_clock_max=max(CLOCKS))
        results["pad_to_next"].append(row)
        print(json.dumps(row), flush=True)
    results["device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    results["gpu_physical_index"] = _PHYS
    out = ROOT / f"reports/segment_xplan_gpu{_PHYS}.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
