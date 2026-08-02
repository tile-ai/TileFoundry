"""Decode Granite-4.0-H-Small on TileFoundry and report how fast it went.

    python run.py --prompt "The capital of France is" --max-new-tokens 32

`--greedy` turns sampling off, which is what you want when comparing token for
token against another implementation; `--seed` fixes the draw when it is on.
`--reference` runs the authored HIR in `model.py` instead of the runtime twin,
which is slow on purpose -- it is the thing the twin is measured against.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=32, metavar="TOKENS")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--greedy", action="store_true", help="take the largest logit instead of sampling"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    # No default: where the weights live is a fact about the machine, not about
    # this model, and a default that only exists on one of them is a guess.
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="the published checkpoint directory (ibm-granite/granite-4.0-h-small)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="run the authored HIR in model.py rather than the runtime twin",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="launch each kernel per step instead of replaying a captured graph",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    import model as sem
    from hf_alias import hf_alias
    from tilefoundry.runtime import SafetensorsResource

    device = (
        str(torch.accelerator.current_accelerator()) if args.device is None else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt))
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_ids = list(getattr(prompt_ids, "ids", prompt_ids))
    if not prompt_ids:
        raise SystemExit("the prompt must encode to at least one token")

    eos = {sem.config.eos_token_id} if isinstance(sem.config.eos_token_id, int) else set(
        sem.config.eos_token_id or ()
    )

    started = perf_counter()
    resource = SafetensorsResource(str(args.ckpt), device=device, alias=hf_alias(sem.config))
    if args.reference:
        runner = sem.Granite4_0_H_Small.load(resource)
    else:
        import runtime_model as rt

        runner = rt.Granite4_0_H_Small()
        runner.load(resource)
    print(f"loaded in {perf_counter() - started:.1f}s", flush=True)

    if args.reference:
        tokens, elapsed = _reference_decode(
            runner, sem, prompt_ids, args, device, eos
        )
    else:
        tokens, elapsed = runner.generate(
            prompt_ids,
            max_new=args.max_new_tokens,
            greedy=args.greedy,
            seed=args.seed,
            temperature=args.temperature,
            eos=eos,
            device=device,
            capture=not args.no_graph,
        )

    text = tokenizer.decode(tokens)
    print(text)
    print(f"{len(tokens) / elapsed:.1f} tok/s")
    return 0


def _reference_decode(loaded, sem, prompt_ids, args, device, eos):
    """The authored Module through its own orchestration methods.

    Sampling is on the host here because the reference has no sampler: it is a
    description of the model, and choosing a token is the caller's step.
    """
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ids = torch.zeros(len(prompt_ids) + args.max_new_tokens, dtype=torch.int64, device=device)
    ids[: len(prompt_ids)] = torch.tensor(prompt_ids, device=device)
    caches = loaded.init_caches(device=device)

    def pick(logits):
        row = logits.reshape(-1).float()
        if args.greedy:
            return int(row.argmax())
        probs = torch.softmax(row / args.temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))

    logits = None
    for step in range(len(prompt_ids)):
        logits, fresh = loaded.forward(
            *loaded.prepare_inputs_for_generation(
                ids[: step + 1], step, caches, device=device
            )
        )
        caches = loaded.append_cache(caches, fresh)

    torch.cuda.synchronize(device)
    started = perf_counter()
    produced = []
    for step in range(args.max_new_tokens):
        token = pick(logits)
        produced.append(token)
        if token in eos:
            break
        at = len(prompt_ids) + step
        ids[at] = token
        logits, fresh = loaded.forward(
            *loaded.prepare_inputs_for_generation(ids[: at + 1], at, caches, device=device)
        )
        caches = loaded.append_cache(caches, fresh)
    torch.cuda.synchronize(device)
    return produced, perf_counter() - started


if __name__ == "__main__":
    raise SystemExit(main())
