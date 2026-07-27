"""Apple AMX compilation capabilities.

Every value is built from the installed ``apple.amx`` document; this module
holds the shape of an AMX architecture value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


@dataclass(frozen=True)
class AppleAmx(Architecture):
    """Apple AMX compilation identity and structural capabilities.

    The register files live here rather than on a device: they are ISA
    geometry, so every part carrying this coprocessor shares them.
    """

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    instruction_capabilities: tuple[str, ...]
    amx_units_per_core: int
    staging_bytes: int
    accumulator_bytes: int

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether AMX has a compute instruction for ``dtype``."""
        return dtype in self.supported_compute_dtypes

    def topology_limit(self, name: str) -> int:
        """Return the structural limit for an AMX topology level."""
        if name == "amx":
            return self.amx_units_per_core
        raise ValueError(
            f"{self.name}: no architecture limit for topology level {name!r}"
        )


__all__ = ["AppleAmx"]
