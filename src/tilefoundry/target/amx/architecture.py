"""Apple AMX compilation capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


@dataclass(frozen=True)
class AppleAmx(Architecture):
    """Apple AMX compilation identity and structural capabilities."""

    name: str = "apple_amx"
    supported_compute_dtypes: tuple[DType, ...] = (
        DType.f16,
        DType.f32,
    )
    instruction_capabilities: tuple[str, ...] = (
        "amx_outer_product",
        "amx_resident_accumulator",
    )
    amx_units_per_core: int = 1

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
