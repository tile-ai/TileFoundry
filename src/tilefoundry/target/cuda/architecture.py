"""SM90 compilation capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType


@dataclass(frozen=True)
class SM90:
    """SM90 compilation identity and structural capabilities."""

    name: str = "sm_90"
    supported_compute_dtypes: tuple[DType, ...] = (
        DType.f32,
        DType.f16,
        DType.bf16,
        DType.fp8e4m3,
    )
    instruction_capabilities: tuple[str, ...] = (
        "tensor_core",
        "wgmma",
        "tma",
    )
    max_threads_per_cta: int = 1024
    max_threads_per_warp: int = 32
    max_warps_per_cta: int = 32

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
