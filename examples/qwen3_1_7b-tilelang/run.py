"""Continue a prompt with Qwen3-1.7B on TileFoundry, and report the rate.

    python run.py --prompt "..." --max-new-tokens 2048

The model is the shipped `qwen3_1_7b` authored HIR (kept verbatim in `ref_src/`,
which is what everything here is measured against); the implementation that runs
is `fast/`, a TileLang runtime twin whose whole decode step is one captured CUDA
graph. See `fast/kernels.py` for the kernels and `fast/engine.py` for why the
step can be captured at all.

The rate covers exactly the steps that produce the continuation. Walking the
prompt is reported separately rather than folded in, because averaging a short
prefill into a long generation flatters the number.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fast"))

NEED_GIB = 12.0


def pick_device(want: str | None) -> str:
    """The emptiest visible accelerator, unless the caller named one.

    Indices come from Torch, not from `nvidia-smi`: this box has several H200s
    and some are in exclusive-process mode with an owner already, so the shell
    restricts what is visible -- and then a global index names the wrong device
    or none at all. Probing through Torch also *tries* each one, which is the
    only way to find out that an exclusive device is already taken.
    """
    if want:
        return want
    import torch

    if not torch.cuda.is_available():
        return str(torch.accelerator.current_accelerator())
    best = None
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue                      # exclusive mode, already owned
        gib = free / 2**30
        if gib >= NEED_GIB and (best is None or gib > best[1]):
            best = (i, gib)
    if best is None:
        raise SystemExit(
            f"no visible accelerator has {NEED_GIB:.0f} GiB free "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
        )
    return f"cuda:{best[0]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="prompt to continue")
    ap.add_argument("--max-new-tokens", type=int, default=2048, metavar="TOKENS")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="published checkpoint directory")
    ap.add_argument("--device", default=None, help="runtime device (default: an idle one)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    import engine as E

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt))
    encoded = tokenizer.encode(args.prompt)
    prompt_ids = list(getattr(encoded, "ids", encoded))
    if not prompt_ids:
        raise SystemExit("decode needs a prompt that encodes to at least one token")

    t0 = perf_counter()
    eng = E.Engine(args.ckpt, HERE / "ref_src", device=device,
                   max_new=args.max_new_tokens, prompt_room=len(prompt_ids) + 8)
    build = perf_counter() - t0

    result = eng.generate(prompt_ids, args.max_new_tokens)
    text = tokenizer.decode(result.tokens)

    print(text)
    print()
    print(f"{result.tokens_per_second:.1f} tok/s "
          f"({len(result.tokens)} tokens in {result.seconds:.3f}s, "
          f"{1e3 * result.seconds / len(result.tokens):.3f} ms/token)")
    print(f"  device {device}  |  prompt {result.prompt_steps} tokens, "
          f"prefill {1e3 * result.prefill_seconds:.1f}ms  |  "
          f"load+compile {build:.1f}s  |  {len(text)} characters generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
