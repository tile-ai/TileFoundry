"""Dump one step's activations, so `tilefoundry check` compares coherent states.

`--inputs random` draws every activation independently, and this model's
activations are not independent: a convolution window, an SSM matrix and a K/V
cache are what previous steps left behind, and three unrelated draws put the
model in a state it can never reach. `tilefoundry check` says so itself and asks
for `--inputs files:...` when the answer has to be decisive.

This advances the op-by-op path for a few steps and writes the resulting
activations, one file per declared non-const parameter, in declared order.

    python dump_acts.py --steps 4 --out acts/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=4, help="steps to advance before dumping")
    ap.add_argument("--out", default="acts")
    a = ap.parse_args()

    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415
    import runtime_model  # noqa: PLC0415
    from model import Nemotron35Lightning30BA3B as SEM  # noqa: PLC0415

    runtime_model.set_impl("ops")
    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=a.device))

    cap = a.steps + 4
    caches = twin.init_caches(device=a.device, capacity=cap)
    ids = torch.full((cap,), 1784, dtype=torch.int64, device=a.device)
    args = None
    for step in range(a.steps):
        args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches,
                                                  device=a.device)
        _, fresh = twin.forward(*args)
        caches = twin.append_cache(caches, fresh)
    args = twin.prepare_inputs_for_generation(ids[: a.steps + 1], a.steps, caches,
                                              device=a.device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = SEM.entry_function().params
    written = []
    for at, p in enumerate(params):
        if p.is_const:
            continue
        # Zero-padded so a lexical sort is the declared order, and named so a
        # file that lands in the wrong slot is visible rather than silent.
        path = out / f"{len(written):03d}_{p.name}.pt"
        torch.save(args[at].detach().cpu(), path)
        written.append(path)
    print(f"wrote {len(written)} activation files to {out}/ after {a.steps} steps")
    print(",".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
