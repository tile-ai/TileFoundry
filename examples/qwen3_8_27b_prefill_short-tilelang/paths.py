"""Where the published checkpoint is. **One place, and no fallback.**

Two directories, and they are not the same thing:

    QWEN38_CKPT    the published `Qwen3.8-27B` checkpoint, as downloaded --
                   `config.json`, `model-*.safetensors`, the tokenizer. Read by
                   the tokenizer, by `tools/hf_reference.py`, by
                   `tools/flops_ledger.py`, by `tools/prepare_ckpt.py` and by
                   the SGLang side of the benchmark.
    ./ckpt         what `tools/prepare_ckpt.py` writes out of it once: the same
                   weights under the canonical names `model.py` declares. The
                   runtime reads this one. It stays inside this directory, so it
                   is not a setting and has no environment variable.

There is deliberately no fallback for `QWEN38_CKPT`: a path that exists on one
machine is not a setting, and defaulting to it behaves like a hardcoded path for
everybody else. The scripts that need it also take `--ckpt`; one of the two ways
is required.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_ENV_CKPT = os.environ.get("QWEN38_CKPT")

CKPT: Path | None = Path(_ENV_CKPT) if _ENV_CKPT else None
PREPARED: Path = ROOT / "ckpt"


def need(which: str = "ckpt", given: str | Path | None = None) -> Path:
    """Resolve a directory, or say exactly what to set.

    `given` is the command line's answer and wins; otherwise the environment.
    Raising here rather than defaulting is the point -- a wrong checkpoint
    silently produces wrong logits, and that is worse than not starting.
    """
    if which == "prepared":
        chosen = Path(given) if given else PREPARED
        if not (chosen / "model.safetensors.index.json").is_file():
            raise SystemExit(
                f"no prepared checkpoint at {chosen}: run `python tools/prepare_ckpt.py` once "
                f"(it reads $QWEN38_CKPT and writes ./ckpt, about 54 GB)"
            )
        return chosen
    if which != "ckpt":
        raise KeyError(which)
    chosen = Path(given) if given else CKPT
    if chosen is None:
        raise SystemExit(
            "no checkpoint directory: pass --ckpt DIR, or set $QWEN38_CKPT.\n"
            "  it is the published Qwen3.8-27B checkpoint as downloaded"
        )
    if not chosen.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {chosen}")
    return chosen


__all__ = ["CKPT", "PREPARED", "ROOT", "need"]
