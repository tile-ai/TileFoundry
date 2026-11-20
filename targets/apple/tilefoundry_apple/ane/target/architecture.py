"""Apple Neural Engine compilation capabilities.

Every value is built from an installed ANE architecture document; this module
holds the shape of an ANE architecture value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import target


@dataclass(frozen=True)
class Architecture(target.Architecture):
    """An ANE generation: the instructions it runs and the tensor extents an instruction accepts."""

    name: str
    capabilities: tuple[str, ...]
    max_channels: int
    max_height: int
    max_width: int
    max_kernel: int


__all__ = ["Architecture"]
