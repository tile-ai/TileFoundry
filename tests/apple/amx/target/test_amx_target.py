"""Pin the AMX target's facts and evidence origins.

Assertions use exact measured device values, including performance-core cache
sizes that unqualified sysctls under-report. They prevent facts being priced from
unmeasured rates or from another hardware level.
"""

from __future__ import annotations

import pytest

from tilefoundry import target
from tilefoundry.ir import expr, types
from tilefoundry_apple import amx


def test_amx_target_reports_and_validates_its_own_topology_levels():
    """Twelve CPU execution positions share AMX capability, not twelve AMX engines."""
    tgt = amx.Target()
    levels = tgt.get_facts(target.TopologyFacts).topologies
    assert tuple(level.name for level in levels) == ("thread",)
    assert not any(level.from_target for level in levels)
    assert tgt.get_facts(target.TopologyLevelFacts, "thread") == target.TopologyLevelFacts("thread", 12, 12)
    assert amx.Target().get_facts(target.TopologyFacts) == tgt.get_facts(target.TopologyFacts)
    assert amx.Target("apple.m2_pro") == tgt
    assert tgt.to_python().text == 'amx.Target("apple.m2_pro")'

    with pytest.raises(ValueError) as topology_error:
        tgt.validate_program_topology(types.Topology("cta", 4))
    assert "unsupported topology level 'cta'" in str(topology_error.value)
    assert "('thread',)" in str(topology_error.value)
    with pytest.raises(target.UnsupportedCapabilityError, match="no Facts projection"):
        tgt.get_facts(target.TopologyLevelFacts, "cta")

    with pytest.raises(ValueError, match="unsupported topology level 'amx'"):
        tgt.validate_program_topology(types.Topology("amx", 1))
    tgt.validate_program_topology(types.Topology("thread", 12))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 12"):
        tgt.validate_program_topology(types.Topology("thread", 13))
    with pytest.raises(ValueError, match="must be positive"):
        tgt.validate_program_topology(types.Topology("thread", 0))
    tgt.validate_program_topology(types.Topology("thread", expr.DimVar("workers", 1, 12)))


def test_amx_values_stand_on_the_installed_documents_and_say_how_they_were_got():
    """Test amx values stand on the installed documents and say how they were got.

    Architecture owns ISA geometry; device owns per-part resources. Every fact
    records provenance and conditions. Vendor bandwidth and a unit count inferred
    from scaling cannot claim measurement, and an absent peak remains absent.
    """
    tgt = amx.Target()
    architecture, device = tgt.architecture, tgt.device

    assert (tgt.architecture_id, tgt.device_id) == ("apple.amx", "apple.m2_pro")

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

    architecture_document = amx.Target.hardware.documents()["apple.amx"]
    device_document = amx.Target.hardware.documents()["apple.m2_pro"]
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
    tgt = amx.Target()
    assert tgt.architecture.supports_compute_dtype(expr.DType.f16)
    assert tgt.device.throughput_for("amx", expr.DType.f32) == 504_900_000_000
    assert tgt.device.throughput_for("neon", expr.DType.f32) == 107_700_000_000
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        tgt.device.throughput_for("amx", expr.DType.f16)
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        tgt.device.throughput_for("sme", expr.DType.f32)
