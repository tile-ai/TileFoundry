"""What a CUDA target tells the algorithms that ask it things.

Each conversion answers exactly one aggregate's question, so what a consumer can
read is visible here rather than spread through the consumers themselves. Nothing
in this module decides anything: it restates the installed hardware documents in
the shape the asking analysis declared.
"""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    ParallelCapacityFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)

from ..base import UnsupportedCapabilityError
from .target import CudaTarget


def memory_hierarchy(target: CudaTarget, query: object = None) -> MemoryHierarchyFacts:
    """The CUDA memory levels and how they are related.

    L1 has no capacity of its own: it and the shared-memory carveout divide one
    physical block per SM, so how much L1 a kernel gets depends on how much
    shared memory it asked for. That is stated as a capacity-sharing edge rather
    than as a number here, because only the program completes it.
    """
    architecture = target.architecture
    device = target.device
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.hbm_capacity_bytes,
                scope="device",
                owner=device.gmem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=architecture.shared_memory_per_cta_bytes,
                scope="cta",
                owner=architecture.smem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=architecture.registers_per_sm_32bit * 4,
                scope="sm",
                owner=architecture.rmem_owner,
            ),
        )
        + (
            (
                ExplicitMemoryLevelFacts(
                    name="tmem",
                    capacity_bytes=architecture.tensor_memory_per_cta_bytes,
                    scope="cta",
                    owner=architecture.tmem_owner,
                ),
            )
            if architecture.tensor_memory_per_cta_bytes is not None
            else ()
        ),
        implicit_levels=(
            ImplicitMemoryLevelFacts(name="l1", capacity_bytes=None, scope="sm"),
            ImplicitMemoryLevelFacts(
                name="l2", capacity_bytes=device.l2_capacity_bytes, scope="device"
            ),
        ),
        relations=(
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l1", far="l2"),
            MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l2", far="gmem"),
            MemoryLevelRelation(
                kind=MemoryRelationKind.SHARES_CAPACITY_WITH,
                near="l1",
                far="smem",
                shared_capacity_bytes=architecture.unified_l1_shared_per_sm_bytes,
            ),
        ),
    )


def _units_per_device(target: CudaTarget, unit: str) -> int:
    """How many of *unit* one card holds, which its peaks are divided among.

    A rate is stated for the whole card, so the rate one unit gets is that
    peak over however many of them the card holds. The card itself is one.
    """
    if unit == "gpu":
        return 1
    if unit == "cta":
        return target.device.sm_count
    if unit == "thread":
        return target.device.sm_count * target.architecture.max_threads_per_cta
    raise UnsupportedCapabilityError(
        f"cuda: no per-unit rate for topology level {unit!r}; the levels this "
        f"target divides its peaks among are ('gpu', 'cta', 'thread')"
    )


def throughput(target: CudaTarget, query: object = None) -> ThroughputFacts:
    """The device rates a roofline divides work by.

    The bandwidth is HBM's, so the memory side of the bound is computed from
    global traffic alone. Shared memory and the register file publish no static
    bandwidth, and inventing one would put a number on the bound that no
    document supports.
    """
    device = target.device
    peaks = tuple(
        sorted(device.dense_flops_per_second.items(), key=lambda item: item[0].name)
    )
    return ThroughputFacts(
        peak_flops_per_second=peaks,
        memory_bandwidth_bytes_per_second=device.hbm_bandwidth_bytes_per_second,
        bandwidth_level="gmem",
    )


def performance_service(
    target: CudaTarget, query: object = None
) -> PerformanceServiceFacts:
    """What one unit of the level asked about gets through, by kind of work.

    The float rates are the device peaks divided among however many of that
    unit the card holds, the same division the roofline's one-unit rates use.
    The services are what the device's own document states, and a device that
    states none prices no work of that kind rather than pricing it at nothing.
    """
    device = target.device
    unit = "cta" if query is None else str(query)
    share = _units_per_device(target, unit)
    return PerformanceServiceFacts(
        unit_flops=tuple(
            (dtype, peak // share)
            for dtype, peak in sorted(
                device.dense_flops_per_second.items(), key=lambda item: item[0].name
            )
        ),
        unit_ops=tuple(
            (kind, rate * target.device.sm_count // share)
            for kind, rate in sorted(device.service_ops_per_second.items())
        ),
        unit_bandwidth=(
            ("gmem", device.hbm_bandwidth_bytes_per_second // share),
        ),
        unit=unit,
    )


def parallel_capacity(
    target: CudaTarget, query: object = None
) -> ParallelCapacityFacts:
    """How many of the level asked about the plan assumes run at once.

    This is a compiler policy, not CUDA's grid limit and not the hardware
    resident-CTA maximum: one active CTA per SM. How many cards run at once is
    not a property of the one card this target describes, so it is refused
    unless the caller stated it.
    """
    unit = "cta" if query is None else str(query)
    if unit == "gpu":
        if target.device_count is None:
            raise UnsupportedCapabilityError(
                "cuda: how many cards run at once is not a property of the card "
                "this target describes; give CudaTarget(device_count=N) to be "
                "measured per gpu"
            )
        return ParallelCapacityFacts(topology="gpu", parallel_units=target.device_count)
    if unit == "thread":
        return ParallelCapacityFacts(
            topology="thread",
            parallel_units=_units_per_device(target, "thread"),
        )
    return ParallelCapacityFacts(topology="cta", parallel_units=target.device.sm_count)


__all__ = [
    "memory_hierarchy",
    "parallel_capacity",
    "performance_service",
    "throughput",
]
