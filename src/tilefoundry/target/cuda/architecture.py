"""SM90 compilation capabilities.

Every value is built from the installed ``nvidia.sm90`` document; this module
holds the shape of an SM90 value, never a copy of its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


@dataclass(frozen=True)
class SM90(Architecture):
    """SM90 compilation identity and structural capabilities.

    The per-SM limits live here rather than on a device: they are properties
    of the microarchitecture, so every product built on SM90 shares them.
    """

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    instruction_capabilities: tuple[str, ...]
    max_threads_per_cta: int
    max_threads_per_warp: int
    max_warps_per_cta: int
    max_resident_ctas_per_sm: int
    shared_memory_per_sm_bytes: int
    shared_memory_per_cta_bytes: int
    # The shared-memory carveout and the L1 data cache divide one physical
    # block, so L1's usable capacity is what a kernel's shared memory leaves of
    # this figure rather than a constant of its own.
    unified_l1_shared_per_sm_bytes: int
    registers_per_sm_32bit: int

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether SM90 has a compute instruction for ``dtype``."""
        return dtype in self.supported_compute_dtypes

    def topology_limit(self, name: str) -> int:
        """Return the structural limit for an SM90 topology level."""
        if name == "thread":
            return self.max_threads_per_cta
        raise ValueError(
            f"{self.name}: no architecture limit for topology level {name!r}"
        )


__all__ = ["SM90"]
