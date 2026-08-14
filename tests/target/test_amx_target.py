"""Pin the AMX target's facts, atom candidates, and evidence origins.

Assertions use exact measured device values, including performance-core cache
sizes that unqualified sysctls under-report. They prevent silent schedules priced
from unmeasured rates or facts belonging to another hardware level.
"""

from __future__ import annotations

import pytest

from tests.fixtures.shapes.matmul_programs import (
    amx_bf16_gemm as bf16_gemm,
)
from tests.fixtures.shapes.matmul_programs import (
    amx_coarse_m_f32_gemm as coarse_m_f32_gemm,
)
from tests.fixtures.shapes.matmul_programs import (
    amx_f32_gemm as f32_gemm,
)
from tests.fixtures.shapes.matmul_programs import (
    amx_odd_m_f32_gemm as odd_m_f32_gemm,
)
from tests.fixtures.shapes.matmul_programs import (
    amx_odd_n_f32_gemm as odd_n_f32_gemm,
)
from tests.fixtures.shapes.matmul_programs import (
    amx_register_sized_f32_gemm as register_sized_f32_gemm,
)
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.schedule.pipeline.facts import PipelineFacts, PipelineFactsQuery
from tilefoundry.target import AmxTarget, TopologyLimitFacts, UnsupportedCapabilityError
from tilefoundry.target.amx.atoms import (
    AMX_REGISTERS,
    CORE_CACHE,
    AMX_FMA32_16x16x1_F32,
    AmxAtom,
    NEON_FMLA_4x4x1_F32,
    candidate_atoms,
)


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


def test_both_catalogue_atoms_are_priced_at_their_own_measured_rates():
    """Price both core units using their own measured rates.

    A 16x16 gemm fitting X/Y/Z registers admits both AMX 16x16 and NEON 4x4
    outer-product atoms. Each field and duration uses that atom's own geometry,
    throughput, and operand-byte cost. Both are arithmetically memory-bound at
    this granularity.
    """
    facts = candidate_atoms(
        register_sized_f32_gemm.entry_function().body,
        register_sized_f32_gemm.resolve_target(),
    )
    assert [fact.atom.op.name for fact in facts] == [
        "AMX_FMA32_16x16x1_F32",
        "NEON_FMLA_4x4x1_F32",
    ]

    amx, neon = facts
    assert isinstance(amx, AtomFact)
    assert isinstance(amx.atom, AmxAtom)
    assert amx.atom.op is AMX_FMA32_16x16x1_F32
    assert amx.atom.op.level is AMX_REGISTERS
    assert amx.shape == (16, 16, 1)
    assert amx.dtype == (DType.f32, DType.f32, DType.f32)
    assert amx.storage == {
        "a_bytes": 64,
        "b_bytes": 64,
        "c_bytes": 1024,
        "operand_bytes": 1152,
    }
    assert amx.resource == {"amx": 1}
    assert amx.is_async is False
    assert amx.compute_duration == pytest.approx(512e9 / 504_900_000_000)
    assert amx.duration == pytest.approx(1152e9 / 200_000_000_000)
    assert amx.duration > amx.compute_duration

    assert neon.atom.op is NEON_FMLA_4x4x1_F32
    assert neon.atom.op.level is CORE_CACHE
    assert neon.shape == (4, 4, 1)
    assert neon.storage == {"a_bytes": 16, "b_bytes": 16, "c_bytes": 64, "operand_bytes": 96}
    assert neon.resource == {"neon": 1}
    assert neon.is_async is False
    assert neon.compute_duration == pytest.approx(32e9 / 107_700_000_000)
    assert neon.duration == pytest.approx(96e9 / 200_000_000_000)


def test_the_hard_filter_is_per_atom_and_covers_storage_shape_and_dtype():
    """Reject atoms independently for storage, shape, and dtype mismatches.

    X/Y/Z geometry rejects a whole 64x128 operand from AMX while cache-streaming
    NEON survives. Shape is evaluated per atom: M=8 fits storage but not AMX's
    16-row granularity, while NEON's 4 divides it. Indivisible extents and bf16
    match neither atom and return an empty list rather than an unsupported query.
    """
    architecture = AmxTarget().architecture
    assert AMX_REGISTERS.budget == (
        ("a_bytes", architecture.staging_bytes),
        ("b_bytes", architecture.staging_bytes),
        ("c_bytes", architecture.accumulator_bytes),
    )
    assert CORE_CACHE.budget == ()
    assert CORE_CACHE.holds({}) and CORE_CACHE.holds({"a_bytes": 10**12})

    whole_tensor = candidate_atoms(f32_gemm.entry_function().body, f32_gemm.resolve_target())
    assert [fact.atom.op.name for fact in whole_tensor] == ["NEON_FMLA_4x4x1_F32"]
    assert not AMX_REGISTERS.holds({"a_bytes": 64 * 128 * 4, "b_bytes": 512, "c_bytes": 512})
    assert CORE_CACHE.holds({"a_bytes": 64 * 128 * 4, "b_bytes": 10**9, "c_bytes": 10**9})

    coarse = candidate_atoms(
        coarse_m_f32_gemm.entry_function().body, coarse_m_f32_gemm.resolve_target()
    )
    assert [fact.atom.op.name for fact in coarse] == ["NEON_FMLA_4x4x1_F32"]
    assert AMX_REGISTERS.holds({"a_bytes": 8 * 8 * 4, "b_bytes": 8 * 16 * 4, "c_bytes": 8 * 16 * 4})

    for indivisible in (odd_m_f32_gemm, odd_n_f32_gemm):
        assert (
            candidate_atoms(indivisible.entry_function().body, indivisible.resolve_target()) == []
        )
    assert candidate_atoms(bf16_gemm.entry_function().body, bf16_gemm.resolve_target()) == []


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


def test_a_core_tile_is_bounded_by_the_l1d_not_by_the_register_files():
    """A core-level tile's resident working set is bounded by the performance core's L1d.

    A core-level tile's resident working set is bounded by the performance
    core's L1d; the register files bound one atom instance instead, which the
    storage filter enforces rather than a per-tile capacity.
    """
    target = AmxTarget()
    facts = target.get_facts(PipelineFacts, PipelineFactsQuery(topology="core", statements=()))
    assert facts.tile_capacity_bytes == target.device.l1d_bytes_per_performance_core
    assert facts.tile_capacity_scope == "core"
    assert target.architecture.accumulator_bytes < (target.device.l1d_bytes_per_performance_core)


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
