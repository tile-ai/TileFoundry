"""CUDA compilation capabilities.

One value type answers for every CUDA architecture. What separates sm_90 from
sm_100 is what their installed documents record, not a class per architecture:
the numbers live in the documents, so a type carrying none of them would only
restate an identity the value already holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


@dataclass(frozen=True)
class CudaArchitecture(Architecture):
    """What one CUDA architecture states about itself.

    The per-SM limits live here rather than on a device: they are properties of
    the microarchitecture, so every product built on it shares them.
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
    # None where the MMA accumulates in registers, so there is no separate store
    # to state a capacity for.
    tensor_memory_per_cta_bytes: int | None

    def _python_import_module(self) -> str:
        # The package re-exports this type, so a rendered constructor imports it
        # from there. A provider's own subclass lives elsewhere and keeps its own
        # module.
        if type(self) is CudaArchitecture:
            return "tilefoundry.target.cuda"
        return super()._python_import_module()

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether this architecture has a compute instruction for ``dtype``."""
        return dtype in self.supported_compute_dtypes

    def topology_limit(self, name: str) -> int:
        """Return the structural limit for a CUDA topology level."""
        if name == "thread":
            return self.max_threads_per_cta
        raise ValueError(
            f"{self.name}: no architecture limit for topology level {name!r}"
        )


__all__ = ["CudaArchitecture"]
