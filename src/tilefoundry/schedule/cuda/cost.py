from __future__ import annotations

from ..cost import CostEstimate
from ..space import EdgeKind, EdgeOption, NodeOption


class CudaCtaCostModel:
    """Small CTA roofline model with target-local calibration defaults."""

    default_bandwidth = 1.0
    default_peak = 1.0
    minimum_share_efficiency = 0.5

    def estimate_node(self, option: NodeOption, context) -> CostEstimate:
        placement = option.placement
        parent_extent = placement.parent_mesh.layout.shape[0]
        share = placement.axis_extents[0] / parent_extent
        efficiency = self.minimum_share_efficiency + (1.0 - self.minimum_share_efficiency) * share
        bandwidth = (context.bandwidth or self.default_bandwidth) * efficiency
        peak = (context.peak_flops or self.default_peak) * efficiency
        duration = max(option.work_bytes / bandwidth, option.flops / peak)
        return CostEstimate(duration=duration, bytes=option.work_bytes, flops=option.flops)

    def estimate_edge(self, option: EdgeOption, context) -> CostEstimate:
        if option.kind is EdgeKind.DIRECT:
            return CostEstimate(duration=0.0, bytes=0, flops=0)
        bandwidth = context.bandwidth or self.default_bandwidth
        duration = option.payload_bytes / bandwidth
        return CostEstimate(duration=duration, bytes=option.payload_bytes, flops=0)


__all__ = ["CudaCtaCostModel"]
