from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CudaArchitectureProfile:
    arch: str
    dense_peak_flops: tuple[tuple[str, float], ...]

    def peak_for(self, dtype: str) -> float:
        for name, value in self.dense_peak_flops:
            if name == dtype:
                return value
        raise KeyError(f"no dense Tensor Core peak for dtype {dtype!r}")


@dataclass(frozen=True, slots=True)
class CudaDeviceProfile:
    name: str
    sm_count: int
    hbm_bandwidth: float


def h200_sxm_profiles(arch: str) -> tuple[CudaArchitectureProfile, CudaDeviceProfile]:
    if arch != "sm_90":
        raise ValueError(f"H200 SXM profile requires arch='sm_90', got {arch!r}")
    # H200 Tensor Core numbers are stored as dense peaks: half of the
    # sparsity-advertised figures used by the hardware specification.
    architecture = CudaArchitectureProfile(
        arch=arch,
        dense_peak_flops=(
            ("bf16", 494.5e12),
            ("f16", 494.5e12),
            ("f32", 67.0e12),
            ("fp8e4m3", 989.0e12),
            ("f8e8m0", 67.0e12),
            ("f4e2m1", 1_978.0e12),
            ("i32", 67.0e12),
            ("i64", 67.0e12),
            ("bool", 67.0e12),
        ),
    )
    device = CudaDeviceProfile(name="h200_sxm", sm_count=132, hbm_bandwidth=4.8e12)
    return architecture, device


__all__ = ["CudaArchitectureProfile", "CudaDeviceProfile", "h200_sxm_profiles"]
