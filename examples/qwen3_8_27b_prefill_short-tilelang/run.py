#!/usr/bin/env python3
"""Prefill one prompt with the tilelang runtime and print its first token.

    python run.py --prompt "..." [--prompt-tokens 1024] [--warmup 3] [--iters 10]

The tokenizer comes from the published checkpoint: pass --ckpt DIR or set
$QWEN38_CKPT (paths.py).

The prompt is encoded without a chat template; --prompt-tokens repeats or cuts the
ids to exactly that many (prompting.py). Weights are read from ./ckpt (written once
by tools/prepare_ckpt.py from the published checkpoint).

Timing: CUDA events around `twin(token_ids, cos, sin)` -- the prefill forward
(input copies into the static graph buffers + graph replay of embedding, 64 layers,
final norm, lm_head). Not inside: weight loading, kernel compilation and graph
capture (the first --warmup call), tokenization, the cos/sin table.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402
import prompting  # noqa: E402
from runtime import engine as eng  # noqa: E402
from runtime.twin import Qwen3_8_27BTwin  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--ckpt", default=None, help="the published checkpoint; or set $QWEN38_CKPT")
    args = ap.parse_args(argv)

    prepared = paths.need("prepared")
    tok = AutoTokenizer.from_pretrained(str(paths.need("ckpt", args.ckpt)))
    ids = prompting.prompt_ids(tok, args.prompt, args.prompt_tokens)
    T = len(ids)
    from runtime import plan
    try:
        plan.for_length(T)
    except ValueError as exc:
        raise SystemExit(f"{exc}: pass --prompt-tokens with one of them (the prompt is {T} tokens)")
    torch.cuda.set_device(0)
    t0 = time.perf_counter()
    twin = Qwen3_8_27BTwin()
    twin.load(SafetensorsResource(str(prepared), device="cuda"))
    token_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
    cos, sin = prompting.rope_tables(T, eng.CFG, "cuda")
    for _ in range(max(1, args.warmup)):
        logits = twin(token_ids, cos, sin)[0]
    torch.cuda.synchronize()
    setup = time.perf_counter() - t0
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        logits = twin(token_ids, cos, sin)[0]
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    first = int(torch.argmax(logits.float()))
    times.sort()
    med = times[len(times) // 2]
    print(f"prompt tokens : {T}")
    print(f"first token   : {first} {tok.decode([first])!r}")
    print(f"prefill       : {med:.3f} ms median of {len(times)} (min {times[0]:.3f}, max {times[-1]:.3f})")
    print(f"throughput    : {T / med * 1e3:,.0f} tok/s")
    print(f"(setup: load + compile + capture {setup:.1f} s, not timed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
