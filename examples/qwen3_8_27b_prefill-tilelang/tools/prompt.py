"""The prompt every measurement and comparison in this directory uses.

`--prompt-tokens N` makes a prompt exactly N tokens long by repeating the
tokenized text and cutting at N. The default text is the published model
card: natural English with code and tables, the same bytes on every run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

#: Importing this module needs the checkpoint, because the default prompt text
#: *is* a file in it. Every script here that measures or compares imports it.
CKPT = paths.need("ckpt")
DEFAULT_TEXT = (CKPT / "README.md").read_text(encoding="utf-8")


def prompt_ids(tokenizer, text: str, n: int | None) -> torch.Tensor:
    """Token ids of *text*, tiled or cut to *n* tokens when *n* is given."""
    ids = tokenizer.encode(text)
    ids = list(getattr(ids, "ids", ids))
    if not ids:
        raise ValueError("the prompt encodes to no tokens")
    if n is not None:
        reps = -(-n // len(ids))
        ids = (ids * reps)[:n]
    return torch.tensor(ids, dtype=torch.int64)
