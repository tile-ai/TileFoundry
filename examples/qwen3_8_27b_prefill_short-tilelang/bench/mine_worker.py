#!/usr/bin/env python3
"""Measurement worker for the tilelang prefill (this round's implementation).

Run with this example's python. Same line protocol as bench/sglang_worker.py:
one JSON command per stdin line, exactly one JSON reply per command on stdout,
all logging on stderr.

  {"cmd":"load"}                                   -> loads the runtime twin + checkpoint
  {"cmd":"bench","token_ids":[...],"warmup":W,"iters":N}
  {"cmd":"quit"}

Timed region (CUDA events on the current stream, synchronize before and after):

    twin(token_ids, cos, sin)

i.e. the runtime twin's `prefill`: copy the three activations into the static
buffers of this length's CUDA graph, replay the graph (embedding, 64 layers, final
norm of the last row, lm_head GEMV). Outside: tokenization (the driver sends ids),
the H2D copy of the ids, building cos/sin (host-side table, as SGLang builds its
own rotary cache at load), compilation and graph capture (first warm-up call of a
length), the NaN poisoning, every assertion.

Recompute evidence, asserted for every warm-up iteration after the first and every
timed iteration: before the call every static buffer of the engine (hidden ping-pong,
norm output, projections, q/k/v/g/beta, attention q, mixer output, MLP activation,
K/V cache, conv windows, recurrent states, last-row norm, logits) is filled with NaN
and checked to be NaN; after the call the logits and all four caches must be finite
and bitwise equal to the first iteration's, and the argmax equal. A graph replay
that skipped any kernel would leave NaN behind (every one of these buffers is
written by the forward before it is read).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_PROTO_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr


def log(*a):
    print("[mine_worker]", *a, file=sys.stderr, flush=True)


def reply(obj):
    _PROTO_OUT.write(json.dumps(obj) + "\n")
    _PROTO_OUT.flush()


def smi(fields, index):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(index)],
                         capture_output=True, text=True, check=True).stdout
    return [x.strip() for x in out.split(",")]


class State:
    twin = None
    torch = None
    phys = None


def physical_index(torch):
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        idx, u = [x.strip() for x in line.split(",")]
        if u.removeprefix("GPU-") == uuid.removeprefix("GPU-"):
            return int(idx)
    return None


def cmd_load(msg):
    import torch

    from runtime.twin import Qwen3_8_27BTwin
    from tilefoundry.runtime import SafetensorsResource

    t0 = time.perf_counter()
    torch.cuda.set_device(0)
    twin = Qwen3_8_27BTwin()
    twin.load(SafetensorsResource(str(ROOT / "ckpt"), device="cuda"))
    State.twin, State.torch = twin, torch
    State.phys = physical_index(torch)
    return {"ok": True, "gpu_name": torch.cuda.get_device_name(0), "gpu_physical_index": State.phys,
            "load_s": time.perf_counter() - t0}


def _buffers(e):
    return [*e.h, e.hmid, e.x, e.proj, e.qkv, e.q, e.k, e.v, e.g, e.beta, e.qa, e.y, e.act,
            e.kc, e.vc, e.conv, e.rec, e.xlast, e.logits, e.ba_part]


def cmd_bench(msg):
    import prompting
    from runtime import engine as eng

    torch = State.torch
    ids = torch.tensor(msg["token_ids"], dtype=torch.int64, device="cuda")
    T = ids.numel()
    cos, sin = prompting.rope_tables(T, eng.CFG, "cuda")
    W, N = int(msg["warmup"]), int(msg["iters"])
    assert W >= 1
    clk_before = smi("clocks.sm,clocks.max.sm,utilization.gpu", State.phys)
    times, argmaxes, clocks, first, asserts = [], [], [], None, []
    walls = []
    for it in range(W + N):
        e = eng._ENGINES.get((T, 0))
        poisoned = e is not None and e.graph is not None
        if poisoned:
            for b in _buffers(e):
                b.fill_(float("nan"))
            torch.cuda.synchronize()
            assert all(bool(torch.isnan(b).all()) for b in _buffers(e)), "poison did not take"
        torch.cuda.synchronize()
        s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        w0 = time.perf_counter()
        s.record()
        logits, kc, vc, conv, rec = State.twin(ids, cos, sin)
        t.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) * 1e3
        ms = s.elapsed_time(t)
        outs = [logits, kc, vc, conv, rec]
        ok_finite = all(bool(torch.isfinite(o).all()) for o in outs)
        am = int(torch.argmax(logits.float()))
        if first is None:
            first = [o.clone() for o in outs]
            first_am = am
        same = all(torch.equal(a, b) for a, b in zip(outs, first))
        rec_assert = {"iter": it, "poisoned_before": poisoned, "finite_after": ok_finite,
                      "bitwise_equal_to_first": same, "argmax_equal": am == first_am}
        if it > 0:
            assert poisoned, "engine buffers were not poisoned"
        assert ok_finite, f"non-finite output after iteration {it}: a kernel did not write its buffer"
        assert same and am == first_am, f"output differs from the first iteration at {it}"
        asserts.append(rec_assert)
        if it >= W:
            times.append(ms)
            walls.append(wall)
            argmaxes.append(am)
            clocks.append(int(smi("clocks.sm", State.phys)[0]))
    clk_after = smi("clocks.sm,clocks.max.sm,utilization.gpu", State.phys)
    st = sorted(times)
    return {"ok": True, "T": T, "times_ms": times, "median_ms": st[len(st) // 2] if len(st) % 2 else
            0.5 * (st[len(st) // 2 - 1] + st[len(st) // 2]), "min_ms": st[0], "max_ms": st[-1],
            "wall_ms": walls, "argmax": argmaxes, "logits_max_abs_diff": 0.0,
            "sm_clock_mhz": {"before": int(clk_before[0]), "after": int(clk_after[0]), "max": int(clk_before[1]),
                             "per_iter_after_forward": clocks},
            "gpu_util_before": int(clk_before[2]), "gpu_physical_index": State.phys,
            "recompute_asserts": {"per_iter": asserts, "all_passed": True,
                                  "method": "NaN-poison all engine buffers before; finite + bitwise-equal outputs after"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if msg["cmd"] == "load":
                reply(cmd_load(msg))
            elif msg["cmd"] == "bench":
                reply(cmd_bench(msg))
            elif msg["cmd"] == "quit":
                reply({"ok": True})
                return
            else:
                reply({"ok": False, "error": f"unknown cmd {msg['cmd']}"})
        except Exception as exc:  # noqa: BLE001
            reply({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
