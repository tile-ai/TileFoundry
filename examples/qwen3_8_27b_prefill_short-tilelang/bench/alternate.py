#!/usr/bin/env python3
"""One session, one card, two workers, measured alternately.

    SGLANG_PYTHON=/path/to/sglang-venv/bin/python \
        python bench/alternate.py --gpu 1 --rounds 6 --warmup 3 --iters 10

Starts bench/sglang_worker.py (in the SGLang virtualenv, $SGLANG_PYTHON) and
bench/mine_worker.py (this virtualenv) on
the same physical card (CUDA_VISIBLE_DEVICES=<gpu> for both), loads both, then for
every length runs `rounds` rounds; each round benches both workers back to back,
alternating which one goes first. Both get the same token ids (the first T tokens of
prompts/prompt.txt, reference/T{T}/token_ids.pt). Only one worker computes at a time.

Rules (also written into the result):
  - per bench call: `warmup` untimed prefills (the first call of a length in my worker
    compiles and captures; SGLang's graphs were captured at load), then `iters` timed;
  - the number per length and side is the median over all timed iterations of all
    rounds; the range (min, max) is over the same set;
  - SM clock: read before and after each call and after every timed iteration.
Writes bench/results/<stamp>.json and prints the table rows.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

#: the python of the SGLang virtualenv -- a different environment from this one
#: (sglang 0.5.21 pins its own torch). There is no default: it is a path on one machine.
SGLANG_PYTHON = os.environ.get("SGLANG_PYTHON")

ROOT = Path(__file__).resolve().parent.parent


class Worker:
    def __init__(self, name, python, script, gpu, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        self.name = name
        self.proc = subprocess.Popen([python, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(log, "w"), text=True, env=env, cwd=ROOT)

    def call(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} died (see its log)")
        out = json.loads(line)
        if not out.get("ok"):
            raise RuntimeError(f"{self.name}: {out.get('error')}\n{out.get('traceback', '')}")
        return out


def others_on_card(gpu):
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
                          capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.splitlines()
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm", "--format=csv,noheader,nounits",
                           "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
    return {"apps": [a for a in apps if a.startswith(uuid)], "util_clock": util}


def wait_quiet(gpu, own_pids, max_s=3600):
    """Other users and other sessions use GPUs 0 and 1 from time to time. Before a call,
    wait until no compute process other than the two workers is on the card; return the
    seconds waited (recorded with the call)."""
    t0 = time.time()
    while True:
        env = others_on_card(gpu)
        foreign = [a for a in env["apps"] if a.split(",")[1].strip() not in own_pids]
        if not foreign or time.time() - t0 > max_s:
            return time.time() - t0, foreign
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mem-fraction", type=float, default=0.45)
    args = ap.parse_args()
    if not SGLANG_PYTHON:
        raise SystemExit("set $SGLANG_PYTHON to the python of a virtualenv with sglang installed")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "bench/results"
    outdir.mkdir(parents=True, exist_ok=True)
    sg = Worker("sglang", Path(SGLANG_PYTHON), ROOT / "bench/sglang_worker.py", args.gpu,
                outdir / f"{stamp}.sglang.log")
    mine = Worker("mine", Path(sys.executable), ROOT / "bench/mine_worker.py", args.gpu,
                  outdir / f"{stamp}.mine.log")
    result = {"stamp": stamp, "gpu_physical_index": args.gpu, "args": vars(args), "rows": {}, "calls": []}
    result["load"] = {"sglang": sg.call({"cmd": "load", "mem_fraction_static": args.mem_fraction}),
                      "mine": mine.call({"cmd": "load"})}
    print("loaded", json.dumps({k: {kk: v[kk] for kk in ("gpu_name", "load_s") if kk in v}
                                for k, v in result["load"].items()}), flush=True)
    for T in args.lengths:
        ids = torch.load(ROOT / f"reference/T{T}/token_ids.pt").tolist()
        ref_first = json.loads((ROOT / f"reference/T{T}/meta.json").read_text())["first_token"]
        per = {"sglang": [], "mine": []}
        clocks = {"sglang": [], "mine": []}
        argm = {"sglang": set(), "mine": set()}
        for r in range(args.rounds):
            order = (sg, mine) if r % 2 == 0 else (mine, sg)
            for w in order:
                waited, foreign = wait_quiet(args.gpu, {str(sg.proc.pid), str(mine.proc.pid)})
                env = others_on_card(args.gpu)
                env["waited_for_quiet_s"] = round(waited, 1)
                env["foreign_after_wait"] = foreign
                out = w.call({"cmd": "bench", "token_ids": ids, "warmup": args.warmup, "iters": args.iters})
                assert out.get("recompute_asserts", {}).get("all_passed", True) is not False
                per[w.name] += out["times_ms"]
                ck = out["sm_clock_mhz"]
                clocks[w.name] += [ck["before"], ck["after"], *ck.get("per_iter_after_forward", [])]
                argm[w.name] |= set(out["argmax"])
                result["calls"].append({"T": T, "round": r, "worker": w.name, "env_before": env,
                                        "median_ms": out["median_ms"], "min_ms": out["min_ms"],
                                        "max_ms": out["max_ms"], "times_ms": out["times_ms"],
                                        "sm_clock_mhz": ck, "argmax": out["argmax"],
                                        "recompute_asserts": out.get("recompute_asserts")})
                print(f"T={T} r{r} {w.name:6s} median {out['median_ms']:.3f} ms "
                      f"[{out['min_ms']:.3f}, {out['max_ms']:.3f}] clk {ck['before']}->{ck['after']}", flush=True)
        row = {}
        for name in ("sglang", "mine"):
            ts = per[name]
            row[name] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts), "n": len(ts),
                         "sm_clock_min": min(clocks[name]), "sm_clock_max": max(clocks[name]),
                         "first_tokens": sorted(argm[name])}
        row["reference_first_token"] = ref_first
        result["rows"][T] = row
        print(f"== T={T}: mine {row['mine']['median_ms']:.3f} ms vs sglang {row['sglang']['median_ms']:.3f} ms; "
              f"first tokens mine {row['mine']['first_tokens']} sglang {row['sglang']['first_tokens']} "
              f"transformers {ref_first}", flush=True)
    for w in (sg, mine):
        try:
            w.call({"cmd": "quit"})
        except Exception:  # noqa: BLE001
            pass
    (outdir / f"{stamp}.json").write_text(json.dumps(result, indent=1))
    print("wrote", outdir / f"{stamp}.json")


if __name__ == "__main__":
    sys.exit(main())
