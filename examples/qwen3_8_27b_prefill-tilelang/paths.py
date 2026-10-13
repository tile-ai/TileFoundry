"""Where the weights live. **One place, and no fallback.**

Two directories, and they are not the same thing:

    QWEN38_CKPT      the published checkpoint, as downloaded -- `config.json`,
                     `model-*.safetensors`, the tokenizer. Everything that runs
                     the model reads this one.
    QWEN38_REFDATA   what `tools/hf_reference.py` writes out of it: one
                     `transformers` forward pass recorded tensor by tensor, plus
                     a small checkpoint of the two layers the per-function checks
                     read. Only the correctness checks need it, it is about
                     20 GB, and it is not shipped -- see README section 4.

There is deliberately no fallback for either: a path that exists on one machine
is not a setting, and defaulting to it behaves like a hardcoded path for
everybody else. Every script that needs one also takes `--ckpt` / `--refdata`,
and one of the two ways is required.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_CKPT = os.environ.get("QWEN38_CKPT")
_ENV_REFDATA = os.environ.get("QWEN38_REFDATA")

CKPT: Path | None = Path(_ENV_CKPT) if _ENV_CKPT else None
REFDATA: Path | None = Path(_ENV_REFDATA) if _ENV_REFDATA else None


def need(which: str, given: str | Path | None = None) -> Path:
    """Resolve one of the two, or say exactly what to set.

    `given` is the command line's answer and wins; otherwise the environment.
    Raising here rather than defaulting is the point -- a wrong checkpoint
    silently produces wrong logits, and that is worse than not starting.
    """
    env, value = {
        "ckpt": ("QWEN38_CKPT", CKPT),
        "refdata": ("QWEN38_REFDATA", REFDATA),
    }[which]
    chosen = Path(given) if given else value
    if chosen is None:
        raise SystemExit(
            f"no {which} directory: pass --{which} DIR, or set ${env}.\n"
            f"  ckpt     = the published Qwen3.8-27B checkpoint as downloaded\n"
            f"  refdata  = what tools/hf_reference.py writes from it"
        )
    if not chosen.is_dir():
        raise SystemExit(f"{which} directory does not exist: {chosen}")
    return chosen


__all__ = ["CKPT", "REFDATA", "need"]
