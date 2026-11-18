"""Apple GPU compilation capabilities.

Every value is built from an installed Metal architecture document; this module
holds the shape of a GPU architecture value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import target


@dataclass(frozen=True)
class Architecture(target.Architecture):
    """An Apple GPU family: the native functions it provides and its SIMD-group width."""

    name: str
    capabilities: tuple[str, ...]
    simd_width: int


__all__ = ["Architecture"]
