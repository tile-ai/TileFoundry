"""Where the weights live. **One place, and no fallback.**

Two directories, and they are not the same thing:

    NEMOTRON35_CKPT      the published checkpoint, as downloaded --
                         `config.json`, `model-*.safetensors`, the tokenizer
    NEMOTRON35_PREPARED  what `prepare_weights.py` writes out of it -- one file
                         per declared weight, already aliased and cast, which is
                         what `tilefoundry check --ckpt DIR` and the twin read

There is deliberately no fallback for either: a path that exists on one machine
is not a setting, and defaulting to it behaves like a hardcoded path for
everybody else. Every script that needs one also takes `--ckpt` / `--prepared`,
and one of the two ways is required.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_CKPT = os.environ.get("NEMOTRON35_CKPT")
_ENV_PREPARED = os.environ.get("NEMOTRON35_PREPARED")

CKPT: Path | None = Path(_ENV_CKPT) if _ENV_CKPT else None
PREPARED: Path | None = Path(_ENV_PREPARED) if _ENV_PREPARED else None


def need(which: str, given: str | Path | None = None) -> Path:
    """Resolve one of the two, or say exactly what to set.

    `given` is the command line's answer and wins; otherwise the environment.
    Raising here rather than defaulting is the point -- a wrong checkpoint
    silently produces wrong tokens, and that is worse than not starting.
    """
    env, value = {
        "ckpt": ("NEMOTRON35_CKPT", CKPT),
        "prepared": ("NEMOTRON35_PREPARED", PREPARED),
    }[which]
    chosen = Path(given) if given else value
    if chosen is None:
        raise SystemExit(
            f"no {which} directory: pass --{which} DIR, or set ${env}.\n"
            f"  ckpt      = the published checkpoint as downloaded\n"
            f"  prepared  = what prepare_weights.py writes from it"
        )
    if not chosen.is_dir():
        raise SystemExit(f"{which} directory does not exist: {chosen}")
    return chosen


__all__ = ["CKPT", "PREPARED", "need"]
