"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry import target
from tilefoundry.ir import expr
from tilefoundry_nvidia import cuda


def test_builtin_targets_own_their_facts_projections() -> None:
    cuda_value = cuda.Target("nvidia.h200_sxm")

    throughput = cuda_value.get_facts(target.ThroughputFacts)
    services = cuda_value.get_facts(target.PerformanceServiceFacts)
    memory = cuda_value.get_facts(target.MemoryHierarchyFacts)

    assert throughput.memory_bandwidth_bytes_per_second == 4_800_000_000_000
    assert dict(services.unit_bandwidth) == {
        "gmem": 4_800_000_000_000 // cuda_value.device.sm_count,
        "smem": 128 * cuda_value.device.sm_clock_hz,
    }
    assert memory.explicit("gmem").capacity_bytes == cuda_value.device.hbm_capacity_bytes
    assert {level.name: level.owner for level in memory.explicit_levels} == {
        "gmem": "gpu",
        "smem": "cta",
        "rmem": "thread",
    }

def test_a_cta_moves_its_bytes_in_flight_per_global_memory_latency() -> None:
    """Little's law: 1024 threads keeping 16 B each in flight move about 46 GB/s."""
    h200, b200 = cuda.Target("nvidia.h200_sxm"), cuda.Target("nvidia.b200_sxm")
    inflight = 1024 * 16
    one = h200.get_facts(target.PerformanceServiceFacts, ("cta", 1, inflight)).bandwidth("gmem")
    assert one == inflight * 1_000_000_000 // 353
    assert 46_000_000_000 < one < 47_000_000_000

    def mesh(tgt, active):
        query = ("cta", active, inflight)
        return tgt.get_facts(target.ThroughputFacts, query).memory_bandwidth_bytes_per_second

    assert mesh(h200, 103) < 4_800_000_000_000 == mesh(h200, 104)
    assert mesh(b200, 1) == b200.device.hbm_bandwidth_bytes_per_second


def test_two_cuda_products_project_the_hardware_each_one_is() -> None:
    """One projection serves both products.

    One projection serves both products, so what separates them is what their
    documents record rather than a branch per architecture.

    Tensor memory is the case that matters: an absent level says the architecture
    has no such store, and a capacity says a CTA's accumulators live in one. A
    projection that reported the same for both would price a Blackwell kernel
    against Hopper's registers.
    """
    hopper = cuda.Target("nvidia.h200_sxm")
    blackwell = cuda.Target("nvidia.b200_sxm")

    assert hopper.get_facts(target.MemoryHierarchyFacts).explicit("tmem") is None
    tmem = blackwell.get_facts(target.MemoryHierarchyFacts).explicit("tmem")
    assert (tmem.capacity_bytes, tmem.scope) == (262_144, "cta")

    throughput = blackwell.get_facts(target.ThroughputFacts)
    peaks = dict(throughput.peak_flops_per_second)
    assert throughput.memory_bandwidth_bytes_per_second == 7_672_320_000_000
    assert peaks[expr.DType.f4e2m1] == 9_000_000_000_000_000
    assert expr.DType.f4e2m1 not in dict(hopper.get_facts(target.ThroughputFacts).peak_flops_per_second)
    assert blackwell.get_facts(target.TopologyFacts).parallel().max_physical_units == 148


def test_a_target_without_a_requested_projection_fails_closed() -> None:
    @dataclass(frozen=True)
    class _UnknownFacts:
        units: int

    with pytest.raises(
        target.UnsupportedCapabilityError, match=r"cuda\.Target: no Facts projection"
    ):
        cuda.Target("nvidia.h200_sxm").get_facts(_UnknownFacts)


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    cuda_value = cuda.Target("nvidia.h200_sxm")

    cuda_levels = cuda_value.get_facts(target.TopologyFacts)
    assert cuda_levels.level("cta") == target.TopologyLevelFacts("cta", None, 132)
    assert cuda_levels.level("thread").max_logical_units == 1024
    assert cuda_levels.parallel() == cuda_levels.level("cta")
    assert cuda_value.topology_limit("cta") == cuda_value.device.sm_count == 132
    assert cuda_value.topology_limit("thread") == cuda_value.architecture.max_threads_per_cta
