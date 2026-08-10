"""Pre-link codegen units (nncase-aligned naming).

A ``LinkableFunction`` is one lowered function's emitted source. A
``LinkableModule`` aggregates a target's linkable functions into that target's
pre-link translation unit (a device ``.cu`` or a host ``.cpp``); the link step
compiles each module with its own toolchain and links them into one
host-callable shared library (a ``LinkedModule``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinkableFunction:
    """One lowered function's pre-link source.

    ``name`` is the function / kernel symbol; ``source`` is its emitted text.
    """

    name: str
    source: str


@dataclass(frozen=True)
class LinkableModule:
    r"""One target's pre-link translation unit.

    ``target`` is the function target name (``cuda`` / ``cpu``); ``language``
    is the source language (``cu`` for an nvcc translation unit, ``cpp`` for a
    plain host translation unit); ``source`` is the assembled translation-unit
    text that the linker compiles; ``functions`` are the constituent
    :class:`LinkableFunction`\\ s carried as metadata.
    """

    target: str
    language: str
    source: str
    functions: tuple[LinkableFunction, ...] = field(default_factory=tuple)


__all__ = ["LinkableFunction", "LinkableModule"]
