"""What each stage of the cooperative kernel costs, by leaving it out.

`nsys` reports one kernel, so a stage's share is not in the trace. This runs
the step with `NEMO_MEGA_SKIP` masking one stage at a time and reports the time
that disappears -- which is the stage's cost including whatever it was hiding.
The masked step computes the wrong answer on purpose; only the clock is read.

    python stage_cost.py [--steps 64]
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import paths

STAGES = [
    (1, "mamba in_proj"), (2, "mamba conv"), (4, "mamba ssm"),
    (8, "mamba gate_norm"), (16, "mamba out_proj"),
    (32, "attn qkv"), (64, "attn scan"), (128, "attn combine"),
    (256, "attn o_proj"),
    (512, "moe logits"), (1024, "moe topk"), (2048, "moe up"),
    (4096, "moe down"), (8192, "moe shared_up"), (16384, "moe finish"),
    (32768, "lm head"),
]

#: A stage's own delta misses what it shares with its neighbours -- the loads
#: one stage leaves in flight are hidden by the next. These are the three
#: branches whole, which is where that shows up.
GROUPS = [
    (1 | 2 | 4 | 8 | 16, "mamba, whole"),
    (32 | 64 | 128 | 256, "attention, whole"),
    (512 | 1024 | 2048 | 4096 | 8192 | 16384, "moe, whole"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args()

    os.environ["NEMO_IMPL"] = "cuda"
    import runtime_model  # noqa: PLC0415
    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415

    dev = "cuda:0"
    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=dev))
    cap = 32 + (len(STAGES) + len(GROUPS) + 3) * (a.steps + 8) + 8
    caches = twin.init_caches(device=dev, capacity=cap)
    ids = torch.full((cap,), 1784, dtype=torch.int64, device=dev)
    step = 0

    def advance(n):
        nonlocal step, caches
        for _ in range(n):
            args = twin.prepare_inputs_for_generation(ids[: step + 1], step,
                                                      caches, device=dev)
            _, fresh = twin.forward(*args)
            caches = twin.append_cache(caches, fresh)
            step += 1

    advance(32)

    def timed() -> float:
        advance(8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        advance(a.steps)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.steps * 1e3

    base = timed()
    print(f"whole step {base:8.3f} ms")
    os.environ["NEMO_MEGA_SKIP"] = str((1 << len(STAGES)) - 1)
    floor = timed()
    os.environ["NEMO_MEGA_SKIP"] = "0"
    print(f"every stage skipped {floor:8.3f} ms  "
          f"({floor / base * 100:.1f}% -- the grid barriers and the launch)")
    rows = []
    for mask, name in STAGES:
        os.environ["NEMO_MEGA_SKIP"] = str(mask)
        rows.append((name, base - timed()))
        os.environ["NEMO_MEGA_SKIP"] = "0"
    for mask, name in GROUPS:
        os.environ["NEMO_MEGA_SKIP"] = str(mask)
        rows.append((name, base - timed()))
        os.environ["NEMO_MEGA_SKIP"] = "0"
    rows.sort(key=lambda r: -r[1])
    print(f"{'stage':18} {'ms':>8} {'share':>8}")
    for name, dt in rows:
        print(f"{name:18} {dt:8.3f} {dt / base * 100:7.1f}%")
    single = sum(dt for name, dt in rows if not name.endswith(", whole"))
    print(f"{'sum of the singles':18} {single:8.3f} {single / base * 100:7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
