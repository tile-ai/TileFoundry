"""Run a shipped causal-LM source directory against its published checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tilefoundry.ir.core.module import Module
from tilefoundry.runtime import SafetensorsResource


def _root(namespace: dict[str, object]) -> Module:
    roots = [
        value
        for value in namespace.values()
        if isinstance(value, Module) and value.target is not None
    ]
    if len(roots) != 1:
        raise SystemExit("model.py must declare exactly one root Module with a target")
    return roots[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path, help="published checkpoint directory")
    parser.add_argument("--prompt", required=True, help="prompt to continue")
    parser.add_argument("--max-new-tokens", required=True, type=int, metavar="TOKENS")
    parser.add_argument(
        "--device",
        help="runtime device (default: Torch's current accelerator)",
    )
    parser.epilog = "Context is limited by the kernel declarations in model.py."
    args = parser.parse_args(argv)

    import model  # noqa: PLC0415
    from generation import decode  # noqa: PLC0415
    from hf_alias import hf_alias  # noqa: PLC0415

    device = str(torch.accelerator.current_accelerator()) if args.device is None else args.device
    loaded = _root(vars(model)).load(
        SafetensorsResource(str(args.ckpt), device=device, alias=hf_alias(model.config))
    )
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    decoded = decode(
        loaded,
        tokenizer,
        args.prompt,
        max_new=args.max_new_tokens,
        device=device,
    )
    print(decoded.text)
    print(f"{decoded.tokens / decoded.seconds:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
