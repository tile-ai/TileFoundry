"""What a CUDA target tells the algorithms that ask it things.

Each conversion answers exactly one aggregate's question, so what a consumer can
read is visible here rather than spread through the consumers themselves. Nothing
in this module decides anything: it restates the installed hardware documents in
the shape the asking analysis declared.
"""

from __future__ import annotations

from tilefoundry import target


def memory_hierarchy(tgt: "Target", query: object = None) -> target.MemoryHierarchyFacts:
    """The CUDA memory levels and how they are related.

    L1 has no capacity of its own: it and the shared-memory carveout divide one
    physical block per SM, so how much L1 a kernel gets depends on how much
    shared memory it asked for. That is stated as a capacity-sharing edge rather
    than as a number here, because only the program completes it.
    """
    architecture = tgt.architecture
    device = tgt.device
    return target.MemoryHierarchyFacts(
        explicit_levels=(
            target.ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.hbm_capacity_bytes,
                scope="device",
                owner=device.gmem_owner,
            ),
            target.ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=architecture.shared_memory_per_cta_bytes,
                scope="cta",
                owner=architecture.smem_owner,
            ),
            target.ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=architecture.registers_per_sm_32bit * 4,
                scope="sm",
                owner=architecture.rmem_owner,
            ),
        )
        + (
            (
                target.ExplicitMemoryLevelFacts(
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
            target.ImplicitMemoryLevelFacts(name="l1", capacity_bytes=None, scope="sm"),
            target.ImplicitMemoryLevelFacts(
                name="l2", capacity_bytes=device.l2_capacity_bytes, scope="device"
            ),
        ),
        relations=(
            target.MemoryLevelRelation(kind=target.MemoryRelationKind.CACHES, near="l1", far="l2"),
            target.MemoryLevelRelation(kind=target.MemoryRelationKind.CACHES, near="l2", far="gmem"),
            target.MemoryLevelRelation(
                kind=target.MemoryRelationKind.SHARES_CAPACITY_WITH,
                near="l1",
                far="smem",
                shared_capacity_bytes=architecture.unified_l1_shared_per_sm_bytes,
            ),
        ),
    )


def card_count(tgt: "Target") -> int:
    """How many cards the deployment runs on; one when nobody said."""
    return 1 if tgt.device_count is None else tgt.device_count


def parallel_units(tgt: "Target", unit: str) -> int:
    """How many of *unit* the deployment runs at once.

    Both the divisor a per-unit rate is the deployment peak over, and the
    answer to how many instances of a level run together -- the same number,
    asked once. A target that never said how many cards is one card.
    """
    cards = card_count(tgt)
    try:
        return {
            "gpu": cards,
            "cta": cards * tgt.device.sm_count,
            "thread": (cards * tgt.device.sm_count * tgt.architecture.max_threads_per_cta),
        }[unit]
    except KeyError:
        raise target.UnsupportedCapabilityError(
            f"cuda: no per-unit rate for topology level {unit!r}; the levels this "
            "target divides its peaks among are ('gpu', 'cta', 'thread')"
        ) from None


def asked(tgt: "Target", query: object) -> tuple[str, int, int | None]:
    """A query's level, running units and bytes in flight; only ``cta`` counts fewer."""
    unit, active, inflight = (*query, None)[:3] if isinstance(query, tuple) else (query, None, None)
    unit = "cta" if unit is None else str(unit)
    share = parallel_units(tgt, unit)
    if active is None or unit != "cta":
        return unit, share, None
    return unit, min(active, share), inflight


def rates(
    tgt: "Target", unit: str, active: int, inflight: int | None = None
) -> target.PerformanceServiceFacts:
    """What *active* units of *unit* get through together, by kind of work.

    The one source of every CUDA rate: the roofline reads it whole and the
    timeline divides it among the *active* units. Float, service and
    shared-memory rates are the deployment's in proportion to the units that
    run. Global memory is the HBM peak, but by Little's law one CTA keeping
    *inflight* bytes in flight moves no more than *inflight* over the device's
    global-memory latency; a device stating no latency has no such cap.
    """
    architecture = tgt.architecture
    device = tgt.device
    cards = card_count(tgt)
    share = parallel_units(tgt, unit)
    gmem = device.hbm_bandwidth_bytes_per_second * cards
    if unit == "cta" and inflight and device.gmem_latency_ns is not None:
        gmem = min(gmem, inflight * active * 1_000_000_000 // device.gmem_latency_ns)
    bandwidth = [("gmem", gmem)]
    if (
        architecture.shared_memory_bandwidth_bytes_per_clock_per_sm is not None
        and device.sm_clock_hz is not None
    ):
        smem = (
            architecture.shared_memory_bandwidth_bytes_per_clock_per_sm
            * device.sm_clock_hz
            * device.sm_count
            * cards
            * active
            // share
        )
        bandwidth.append(("smem", smem))
    return target.PerformanceServiceFacts(
        unit_flops=tuple(
            (dtype, peak * cards * active // share)
            for dtype, peak in sorted(
                device.dense_flops_per_second.items(), key=lambda item: item[0].name
            )
        ),
        unit_ops=tuple(
            (kind, rate * device.sm_count * cards * active // share)
            for kind, rate in sorted(device.service_ops_per_second.items())
        ),
        unit_bandwidth=tuple(bandwidth),
        unit=unit,
    )


def throughput(tgt: "Target", query: object = None) -> target.ThroughputFacts:
    """The rates a roofline divides the whole program's work by: `rates`, whole.

    Asked about no level, the whole deployment; the bound reads global traffic.
    """
    whole = rates(tgt, *asked(tgt, query))
    return target.ThroughputFacts(
        peak_flops_per_second=whole.unit_flops,
        memory_bandwidth_bytes_per_second=whole.bandwidth("gmem"),
        bandwidth_level="gmem",
    )


def performance_service(tgt: "Target", query: object = None) -> target.PerformanceServiceFacts:
    """What one unit of the level asked about gets through: `rates` among its active units."""
    unit, active, inflight = asked(tgt, query)
    whole = rates(tgt, unit, active, inflight)
    return target.PerformanceServiceFacts(
        unit_flops=tuple((dtype, rate // active) for dtype, rate in whole.unit_flops),
        unit_ops=tuple((kind, rate // active) for kind, rate in whole.unit_ops),
        unit_bandwidth=tuple((level, rate // active) for level, rate in whole.unit_bandwidth),
        unit=unit,
    )


__all__ = [
    "memory_hierarchy",
    "parallel_units",
    "performance_service",
    "rates",
    "throughput",
]
