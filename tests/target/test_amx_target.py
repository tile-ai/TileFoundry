"""Pin the AMX target's facts and evidence origins.

Assertions use exact measured device values, including performance-core cache
sizes that unqualified sysctls under-report. They prevent facts being priced from
unmeasured rates or from another hardware level.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import AmxTarget, TopologyLimitFacts, UnsupportedCapabilityError


def test_amx_target_reports_and_validates_its_own_topology_levels():
    """Test amx target reports and validates its own topology levels.

    The AMX levels are the core a tile stream runs on and the AMX unit inside
    it that issues one atom. The core limit is the measured performance-core
    count; the unit limit comes from the architecture.

    Each limit is projected as a Target Fact. A declared extent is validated
    against its level's own limit, and an AMX core count is always static: there
    is no launch shape to defer it to, so a symbolic extent is refused rather
    than accepted and counted later.
    """
    target = AmxTarget()
    assert target.topology_levels == ("core", "amx")
    assert target.get_facts(TopologyLimitFacts, "core").max_static_extent == 8
    assert target.get_facts(TopologyLimitFacts, "amx").max_static_extent == 1
    assert AmxTarget().topology_levels == target.topology_levels
    assert AmxTarget("apple.m2_pro") == target
    assert target.to_python().text == 'AmxTarget("apple.m2_pro")'

    with pytest.raises(ValueError) as topology_error:
        target.validate_program_topology(Topology("cta", 4))
    assert "unsupported topology level 'cta'" in str(topology_error.value)
    assert "('core', 'amx')" in str(topology_error.value)
    with pytest.raises(UnsupportedCapabilityError, match="no Facts projection"):
        target.get_facts(TopologyLimitFacts, "cta")

    target.validate_program_topology(Topology("core", 8))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 8"):
        target.validate_program_topology(Topology("core", 9))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("core", 0))
    target.validate_program_topology(Topology("core", DimVar("cores", 1, 8)))


def test_amx_values_stand_on_the_installed_documents_and_say_how_they_were_got():
    """Test amx values stand on the installed documents and say how they were got.

    Architecture owns ISA geometry; device owns per-part resources. Every fact
    records provenance and conditions. Vendor bandwidth and a unit count inferred
    from scaling cannot claim measurement, and an absent peak remains absent.
    """
    target = AmxTarget()
    architecture, device = target.architecture, target.device

    assert (target.architecture_id, target.device_id) == ("apple.amx", "apple.m2_pro")

    assert architecture.staging_bytes == 512
    assert architecture.accumulator_bytes == 4096
    for moved in ("amx_staging_bytes", "amx_accumulator_bytes"):
        assert not hasattr(device, moved)

    assert device.l1d_bytes_per_performance_core == 128 * 1024
    assert device.l1d_bytes_per_efficiency_core == 64 * 1024
    assert device.l2_bytes_per_performance_cluster == 16 * 1024 * 1024
    assert device.l2_bytes_per_efficiency_cluster == 4 * 1024 * 1024
    assert device.cache_line_bytes == 128
    assert device.performance_core_count == 8
    assert device.sm_count == 2

    architecture_document = AmxTarget.hardware.documents()["apple.amx"]
    device_document = AmxTarget.hardware.documents()["apple.m2_pro"]
    assert device_document.fact("memory.unified.bandwidth").origin == "vendor"
    assert device_document.fact("compute.amx_unit_count").origin == "estimated"
    assert device_document.fact("throughput.amx.f32").origin == "measured"
    assert device_document.fact("throughput.neon.f32").origin == "measured"
    assert device_document.fact("geometry.neon_f32_outer_product").origin == "derived"
    assert not architecture_document.fact("throughput.f32_instruction_peak").available

    for document in (architecture_document, device_document):
        for fact in document.facts.values():
            assert fact.conditions


def test_unmeasured_units_and_dtypes_have_no_throughput_entry():
    """FMA16 exists in the instruction set, but no f16 rate was measured on either unit.

    FMA16 exists in the instruction set, but no f16 rate was measured on
    either unit, so asking for one raises instead of returning a guess -- as does
    naming a unit that has no recorded rate at all.
    """
    target = AmxTarget()
    assert target.architecture.supports_compute_dtype(DType.f16)
    assert target.device.throughput_for("amx", DType.f32) == 504_900_000_000
    assert target.device.throughput_for("neon", DType.f32) == 107_700_000_000
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        target.device.throughput_for("amx", DType.f16)
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        target.device.throughput_for("sme", DType.f32)
