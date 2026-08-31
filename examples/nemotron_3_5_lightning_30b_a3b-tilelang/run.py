"""Continue a prompt with Nemotron-3.5-Lightning-30B-A3B, and say how fast.

    python run.py --prompt "..." --max-new-tokens 512 [--greedy] [--seed N]

Every token -- the prompt's as well as the continuation's -- goes through the
same one-launch decode step; there is no separate prefill path. The tok/s
printed is decode only: the clock starts once the prompt has been consumed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tilefoundry.runtime import SafetensorsResource

import paths


def sampler_for(args):
    """Greedy, or the published generation config's temperature and top-p."""
    if args.greedy:
        def greedy(logits):
            return int(torch.argmax(logits).item())
        return greedy

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    temperature, top_p = args.temperature, args.top_p

    def sample(logits):
        probs = torch.softmax(logits.reshape(-1).float() / temperature, dim=-1)
        order = torch.argsort(probs, descending=True)
        ordered = probs[order]
        keep = int(torch.searchsorted(torch.cumsum(ordered, 0), top_p).item()) + 1
        head = ordered[:keep] / ordered[:keep].sum()
        pick = int(torch.multinomial(head.cpu(), 1, generator=generator).item())
        return int(order[pick].item())

    return sample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True, metavar="TOKENS")
    ap.add_argument("--greedy", action="store_true", help="take the largest logit")
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ckpt", default=None, help="published checkpoint, for its tokenizer"
                                            " (or $NEMOTRON35_CKPT)")
    ap.add_argument("--prepared", default=None, help="prepared weight directory"
                                                " (or $NEMOTRON35_PREPARED)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--impl", default=None, choices=["mega", "ops"])
    ap.add_argument("--capacity", type=int, default=None,
                    help="K/V capacity to allocate (default: max_position_embeddings)")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    args = ap.parse_args(argv)

    if args.impl:
        os.environ["NEMO_IMPL"] = args.impl
    import runtime_model  # noqa: PLC0415 -- reads NEMO_IMPL at import
    from generation import decode  # noqa: PLC0415

    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(str(paths.need("prepared", args.prepared)),
                                  device=args.device))
    tokenizer = AutoTokenizer.from_pretrained(paths.need("ckpt", args.ckpt))
    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    decoded = decode(twin, tokenizer, prompt, max_new=args.max_new_tokens,
                     sampler=sampler_for(args), eos=(2, 11), device=args.device)
    print(decoded.text)
    print(f"\n{decoded.tokens} tokens in {decoded.seconds:.3f} s"
          f"  ->  {decoded.tokens / decoded.seconds:.1f} tok/s"
          f"   (prompt {decoded.prompt_steps} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
