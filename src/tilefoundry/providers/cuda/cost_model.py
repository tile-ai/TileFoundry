from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.schedule.cost import CostEstimate

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
        rounded = int(round(seconds * 1e9))
        return max(1 if seconds > 0 else 0, rounded)

    def reshard_duration_ns(self, *, total_read_write_bytes: float, cta_count: int) -> int:
        if cta_count <= 0 or cta_count > self.device.sm_count:
            raise ValueError(f"CTA share must use 1..{self.device.sm_count} CTAs")
        share = cta_count / self.device.sm_count
        seconds = total_read_write_bytes / (self.device.hbm_bandwidth * share)
        rounded = int(round(seconds * 1e9))
        return self.fixed_latency_ns + max(1 if seconds > 0 else 0, rounded)

    def estimate_node(self, option, context) -> CostEstimate:
        work = option.candidate.estimated_work
        share = option.candidate.cta_count / self.device.sm_count
        peak = self.architecture.peak_for(work.dtype) * share
        compute_ns = int(round(work.flops / peak * 1e9)) if peak else 0
        memory_ns = int(round(work.traffic_bytes / (self.device.hbm_bandwidth * share) * 1e9))
        if work.flops > 0:
            compute_ns = max(1, compute_ns)
        if work.traffic_bytes > 0:
            memory_ns = max(1, memory_ns)
        return CostEstimate(
            duration_ns=max(compute_ns, memory_ns),
            traffic_bytes=int(work.traffic_bytes),
            flops=int(work.flops),
            compute_time_ns=compute_ns,
            memory_time_ns=memory_ns,
        )

    def estimate_edge(self, option, context) -> CostEstimate:
        if option.kind.value == "direct":
            return CostEstimate(duration_ns=0)
        duration = self.reshard_duration_ns(
            total_read_write_bytes=option.payload_bytes,
            cta_count=option.cta_count,
        )
        return CostEstimate(duration_ns=duration, traffic_bytes=option.payload_bytes)


__all__ = ["CudaFormulaCostModel"]
