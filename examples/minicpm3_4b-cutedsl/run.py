"""Decode MiniCPM3-4B on TileFoundry, and say how fast.

    python run.py --prompt "The capital of France is" --max-new-tokens 32

Options:
    --seed N        seed the sampler
    --greedy        take the argmax instead of sampling, for comparing runs
    --backend B     which kernels: `cute` (default) or `torch` (the reference
                    the CuTeDSL kernels were written against)
    --reference     run the authored Module through the HIR evaluator instead
                    of its runtime twin -- the same tokens, very slowly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".venv/share/tilefoundry/orchestrator/causal_lm"))

#: Where the prepared checkpoint is written, beside this file. The *published*
#: checkpoint has no default: that is a fact about the machine, and a default
#: which exists on only one of them is a guess. `--ckpt` states it.
PREPARED = HERE / "prepared"


def _sampler(greedy: bool, temperature: float, top_p: float):
    """`generation_config.json`: sampling at T=0.8, top-p 0.8, unless told not to."""
    import torch

    from generation import greedy as argmax  # type: ignore

    if greedy:
        return argmax

    def sample(logits) -> int:
        row = logits.reshape(-1).float() / temperature
        probs = torch.softmax(row, dim=-1)
        order = torch.argsort(probs, descending=True)
        ranked = probs[order]
        keep = torch.cumsum(ranked, dim=-1) - ranked < top_p
        ranked = torch.where(keep, ranked, torch.zeros_like(ranked))
        return int(order[torch.multinomial(ranked, 1)].item())

    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--backend", default="cute", choices=("cute", "torch"))
    ap.add_argument("--reference", action="store_true",
                    help="run the authored Module through the evaluator instead")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--prepared", default=str(PREPARED))
    ap.add_argument("--ckpt", required=True,
                    help="the published checkpoint directory (openbmb/MiniCPM3-4B)")
    ap.add_argument("--quiet", action="store_true", help="print the two lines only")
    args = ap.parse_args()

    import torch
    from tokenizers import Tokenizer

    from generation import decode  # type: ignore

    if not (Path(args.prepared) / "model.safetensors.index.json").exists():
        # The published checkpoint is one `pytorch_model.bin` under names the
        # authored Module does not use. Converting it is `Module.prepare`'s job
        # and it is done once; doing it here means this entry point needs
        # nothing but the checkpoint.
        print(f"# no prepared checkpoint at {args.prepared}; converting "
              f"{args.ckpt} (about a minute, once)")
        import subprocess

        subprocess.run(
            [sys.executable, str(HERE / "tools" / "prep_ckpt.py"),
             "--ckpt", args.ckpt, "--out", args.prepared],
            check=True,
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda"
    tokenizer = Tokenizer.from_file(str(Path(args.ckpt) / "tokenizer.json"))

    if args.reference:
        from tilefoundry.runtime.resource import SafetensorsResource

        from model import MiniCPM3_4B
        loaded = MiniCPM3_4B.load(SafetensorsResource(args.prepared, device=device))
        what = "authored Module (HIR evaluator)"
    else:
        import kernels

        kernels.select(args.backend)
        import runtime_model

        loaded = runtime_model.load(args.prepared, device=device)
        what = f"runtime twin ({kernels.active()} kernels)"

    with torch.inference_mode():
        out = decode(
            loaded, tokenizer, args.prompt,
            max_new=args.max_new_tokens,
            sampler=_sampler(args.greedy, args.temperature, args.top_p),
            eos=tuple(_eos_ids(args.ckpt)), device=device,
        )

    if not args.quiet:
        print(f"# {what}, {out.prompt_steps} prompt tokens, "
              f"{'greedy' if args.greedy else 'sampled'}")
        print(f"prompt: {args.prompt}")
    print(out.text)
    print(f"{out.tokens} tokens in {out.seconds:.4f}s = "
          f"{out.tokens / out.seconds:.2f} tokens/s")


def _eos_ids(ckpt):
    import json

    gen = json.loads((Path(ckpt) / "generation_config.json").read_text())
    eos = gen.get("eos_token_id", [])
    return eos if isinstance(eos, list) else [eos]


if __name__ == "__main__":
    main()
