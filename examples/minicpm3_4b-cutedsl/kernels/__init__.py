"""The decode path's kernels.

Two implementations of one interface (`reference_ops`' names and signatures):

- `reference_ops` -- torch, the thing a kernel is first written against
- `cute_ops`      -- CuTeDSL, the thing that actually runs

`select(backend)` swaps which one this module's names point at, so a single
kernel can be moved across and the rest of the path stays where it was. That is
what makes a disagreement have one suspect.
"""
from __future__ import annotations

from . import reference_ops

_NAMES = tuple(reference_ops.__all__)

BACKENDS = ("torch", "cute")

#: Whether a backend's ops can run inside a CUDA graph capture. The torch
#: spelling cannot: `rope_attend` reads the context length with `int(ctx)` and
#: the embedding indexes with a device tensor, both of which synchronise, and a
#: capturing stream forbids that. It is the reference, not the fast path, so it
#: keeps the readable spelling and the twin simply does not capture it.
CAPTURABLE = {"torch": False, "cute": True}

_active = "torch"


def select(backend: str) -> None:
    """Point this module's kernel names at *backend*'s implementations.

    `"cute"` falls back per name: an op the CuTeDSL side does not define yet
    keeps the torch one, so the path runs from the first kernel onwards.
    """
    global _active
    if backend not in BACKENDS:
        raise ValueError(f"unknown kernel backend {backend!r}; have {BACKENDS}")
    source = reference_ops
    if backend == "cute":
        from . import cute_ops
        source = cute_ops
    for name in _NAMES:
        globals()[name] = getattr(source, name, getattr(reference_ops, name))
    _active = backend


def active() -> str:
    return _active


def capturable() -> bool:
    """Can a step built from the active backend be recorded as a CUDA graph?"""
    return CAPTURABLE[_active]


select("torch")
