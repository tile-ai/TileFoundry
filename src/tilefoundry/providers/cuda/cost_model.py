from __future__ import annotations

from dataclasses import dataclass

from .profiles import CudaArchitectureProfile, CudaDeviceProfile


@dataclass(frozen=True, slots=True)
class CudaFormulaCostModel:
    architecture: CudaArchitectureProfile
    device: CudaDeviceProfile
    fixed_latency_ns: int = 0

    def duration_ns(
        self,
        *,
        flops: float,
        traffic_bytes: float,
        dtype: str,
        cta_count: int,
    ) -> int:
        if cta_count <= 0 or cta_count > self.device.sm_count:
            raise ValueError(f"CTA share must use 1..{self.device.sm_count} CTAs")
        share = cta_count / self.device.sm_count
        peak = self.architecture.peak_for(dtype) * share
        seconds = max(flops / peak, traffic_bytes / (self.device.hbm_bandwidth * share))
        return max(0, int(round(seconds * 1e9)))

    def reshard_duration_ns(self, *, total_read_write_bytes: float, cta_count: int) -> int:
        if cta_count <= 0 or cta_count > self.device.sm_count:
            raise ValueError(f"CTA share must use 1..{self.device.sm_count} CTAs")
        share = cta_count / self.device.sm_count
        seconds = total_read_write_bytes / (self.device.hbm_bandwidth * share)
        return self.fixed_latency_ns + max(0, int(round(seconds * 1e9)))


__all__ = ["CudaFormulaCostModel"]
