"""The AMX target's own facts: its topology levels, its atom candidate
enumeration and the recorded origin of every fact it stands on.

The device facts are the ones actually measured on the host this target
describes, so the assertions below name the exact values rather than a range --
in particular the performance-core cache sizes, which the unqualified
``hw.l1dcachesize`` / ``hw.l2cachesize`` sysctls under-report by naming the
efficiency core instead.
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import Schedule
from tilefoundry.schedule.facts import AtomFact, TileStoreFacts
from tilefoundry.target import AmxTarget, resolve_target
from tilefoundry.target.amx.atoms import (
    AMX_REGISTERS,
    CORE_CACHE,
    AMX_FMA32_16x16x1_F32,
    AmxAtom,
    NEON_FMLA_4x4x1_F32,
    candidate_atoms,
)
from tilefoundry.target.facts import TARGET_FACTS
from tilefoundry.target.hardware import HARDWARE_SPECS


@func(target="amx")
def f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def register_sized_f32_gemm(
    x: Tensor[(16, 8), "f32"],
    w: Tensor[(8, 16), "f32"],
) -> Tensor[(16, 16), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def coarse_m_f32_gemm(
    x: Tensor[(8, 8), "f32"],
    w: Tensor[(8, 16), "f32"],
) -> Tensor[(8, 16), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def odd_m_f32_gemm(
    x: Tensor[(18, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(18, 64), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def odd_n_f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 18), "f32"],
) -> Tensor[(64, 18), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def bf16_gemm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)  # noqa: F405
    return h


# ---------------------------------------------------------------------------
# AC-4-1
# ---------------------------------------------------------------------------


def test_amx_target_reports_its_own_topology_levels():
    """The AMX levels are the core a tile stream runs on and the AMX unit inside
    it that issues one atom. The core limit is the measured performance-core
    count; the unit limit comes from the architecture."""
    target = AmxTarget()
    assert target.topology_levels == ("core", "amx")
    assert target.topology_limit("core") == 8
    assert target.topology_limit("amx") == 1
    assert resolve_target("amx").topology_levels == target.topology_levels


def test_an_unsupported_topology_level_names_the_supported_ones():
    """AC-4-1: asking for a level the target does not have raises, and the
    message lists the levels it does -- for both the limit lookup and the
    declared-topology validation."""
    target = AmxTarget()
    with pytest.raises(ValueError) as limit_error:
        target.topology_limit("cta")
    with pytest.raises(ValueError) as topology_error:
        target.validate_program_topology(Topology("cta", 4))

    for error in (limit_error, topology_error):
        message = str(error.value)
        print("\n===", message[-120:])
        assert "unsupported topology level 'cta'" in message
        assert "('core', 'amx')" in message


def test_a_core_extent_past_the_performance_core_count_is_rejected():
    """A declared extent is validated against the level's own limit, and an AMX
    core count is always static -- there is no launch shape to defer it to."""
    target = AmxTarget()
    target.validate_program_topology(Topology("core", 8))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 8"):
        target.validate_program_topology(Topology("core", 9))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("core", 0))
    with pytest.raises(ValueError, match="requires a positive static integer"):
        target.validate_program_topology(Topology("core", DimVar("cores", 1, 8)))


def test_each_amx_level_service_is_bound_exactly_once():
    """The core stage binds one Schedule service; a stage the target does not
    serve raises. Hardware facts are projected rather than served, so there is
    no second service to bind."""
    target = AmxTarget()
    assert target.service(Schedule, "core").stage == "core"
    with pytest.raises(ValueError, match="expected exactly one service"):
        target.service(Schedule, "cta")


# ---------------------------------------------------------------------------
# AC-4-2
# ---------------------------------------------------------------------------


def test_a_register_sized_f32_gemm_lists_the_amx_outer_product_atom():
    """AC-4-2: M=16, N=16 both divide the atom's 16x16 shape, K divides its own
    extent of one, and the whole gemm's three operands fit the X/Y/Z register
    files the atom addresses them as. Every field is checked against the atom's
    real numbers, not a placeholder."""
    facts = candidate_atoms(register_sized_f32_gemm.entry_function().body, register_sized_f32_gemm.resolve_target())
    print("\n=== candidate AtomFacts (f32 gemm, M=16 N=16 K=8) ===")
    for fact in facts:
        print(fact)

    assert [fact.atom.op.name for fact in facts] == [
        "AMX_FMA32_16x16x1_F32",
        "NEON_FMLA_4x4x1_F32",
    ]
    fact = facts[0]
    assert isinstance(fact, AtomFact)
    assert fact.shape == (16, 16, 1)
    assert fact.dtype == (DType.f32, DType.f32, DType.f32)
    assert isinstance(fact.atom, AmxAtom)
    assert fact.atom.op is AMX_FMA32_16x16x1_F32
    assert fact.atom.op.level is AMX_REGISTERS

    # One 16x16 f32 outer product: a 16-element A operand in X, a 16-element B
    # operand in Y and the 16x16 f32 accumulator tile it lands in, in Z.
    assert fact.storage == {
        "a_bytes": 64,
        "b_bytes": 64,
        "c_bytes": 1024,
        "operand_bytes": 1152,
    }
    assert fact.resource == {"amx": 1}
    assert fact.is_async is False

    # 2*16*16*1 flops over the measured per-unit AMX f32 rate; 1152 bytes over
    # the unified-memory bandwidth. Memory-bound at this granularity.
    assert fact.compute_duration == pytest.approx(512e9 / 504_900_000_000)
    assert fact.duration == pytest.approx(1152e9 / 200_000_000_000)
    assert fact.duration > fact.compute_duration


def test_the_neon_atom_is_a_candidate_for_the_same_gemm_at_its_own_rate():
    """The second catalogue entry: a 4x4 f32 outer product on the core's NEON
    pipes, its operands streamed through cache rather than held in registers,
    priced at the measured NEON rate instead of the AMX one."""
    facts = candidate_atoms(register_sized_f32_gemm.entry_function().body, register_sized_f32_gemm.resolve_target())
    fact = facts[1]

    assert fact.atom.op is NEON_FMLA_4x4x1_F32
    assert fact.shape == (4, 4, 1)
    assert fact.atom.op.level is CORE_CACHE
    assert fact.storage == {"a_bytes": 16, "b_bytes": 16, "c_bytes": 64, "operand_bytes": 96}
    assert fact.resource == {"neon": 1}
    assert fact.is_async is False

    # 2*4*4*1 flops over the measured NEON f32 rate; 96 bytes over the
    # unified-memory bandwidth.
    assert fact.compute_duration == pytest.approx(32e9 / 107_700_000_000)
    assert fact.duration == pytest.approx(96e9 / 200_000_000_000)


def test_operands_too_big_for_the_register_files_leave_only_the_neon_atom():
    """The storage filter is what separates the two units: this gemm's M/N/K all
    divide the AMX shape, but a 64x128 f32 A operand is 32 KiB against 512 B of
    X, so only the cache-streaming NEON atom survives. That is why an untiled
    whole-tensor statement lands on SIMD."""
    facts = candidate_atoms(f32_gemm.entry_function().body, f32_gemm.resolve_target())
    print("\n=== candidates (f32 gemm, M=64 N=64 K=128) ===", [f.atom.op.name for f in facts])

    assert [fact.atom.op.name for fact in facts] == ["NEON_FMLA_4x4x1_F32"]
    assert not AMX_REGISTERS.holds({"a_bytes": 64 * 128 * 4, "b_bytes": 512, "c_bytes": 512})
    assert CORE_CACHE.holds({"a_bytes": 64 * 128 * 4, "b_bytes": 10**9, "c_bytes": 10**9})


@pytest.mark.parametrize("fn", [odd_m_f32_gemm, odd_n_f32_gemm])
def test_an_indivisible_extent_lists_no_candidate(fn):
    """AC-4-2: an extent of 18 is a whole multiple of neither atom's M or N
    (16 for AMX, 4 for NEON) -- hard-filtered out, an empty list rather than an
    error."""
    assert candidate_atoms(fn.entry_function().body, fn.resolve_target()) == []


def test_an_extent_coarser_than_one_atom_only_lists_the_finer_one():
    """The shape filter is per atom, not per catalogue: M=8 fits the register
    files whole, so nothing about storage rules the AMX atom out -- its own 16
    row granularity does, while NEON's 4 divides 8."""
    facts = candidate_atoms(coarse_m_f32_gemm.entry_function().body, coarse_m_f32_gemm.resolve_target())

    assert [fact.atom.op.name for fact in facts] == ["NEON_FMLA_4x4x1_F32"]
    assert AMX_REGISTERS.holds({"a_bytes": 8 * 8 * 4, "b_bytes": 8 * 16 * 4, "c_bytes": 8 * 16 * 4})


def test_a_non_f32_gemm_lists_no_candidate():
    """Both modelled atoms are f32, so a bf16 gemm's operand dtypes do not match
    any registered atom."""
    assert candidate_atoms(bf16_gemm.entry_function().body, bf16_gemm.resolve_target()) == []


def test_candidate_atoms_defaults_and_resolves_a_backend_name():
    """``target=None`` defaults to the AMX target and a ``target=`` string is
    resolved, matching ``@func(target="amx")``'s own surface."""
    explicit = candidate_atoms(f32_gemm.entry_function().body, AmxTarget())
    assert candidate_atoms(f32_gemm.entry_function().body) == explicit
    assert candidate_atoms(f32_gemm.entry_function().body, "amx") == explicit


def test_a_non_amx_target_raises():
    """The per-atom device facts this bridge reads (per-unit AMX and NEON
    throughput, unified-memory bandwidth) exist on the AMX device only."""
    with pytest.raises(NotImplementedError, match="only AmxTarget is supported"):
        candidate_atoms(f32_gemm.entry_function().body, target="cuda")


# ---------------------------------------------------------------------------
# AC-4-3
# ---------------------------------------------------------------------------

def test_amx_values_stand_on_the_installed_documents():
    """The architecture carries the ISA geometry and the device the per-part
    resources, each built from its own installed document."""
    target = AmxTarget()
    architecture, device = target.architecture, target.device

    assert (target.architecture_id, target.device_id) == ("apple.amx", "apple.m2_pro")
    # ISA geometry belongs to the coprocessor, not to the package carrying it.
    assert architecture.staging_bytes == 512
    assert architecture.accumulator_bytes == 4096
    for moved in ("amx_staging_bytes", "amx_accumulator_bytes"):
        assert not hasattr(device, moved)

    # The performance core's caches, not the efficiency core's 64 KiB / 4 MiB.
    assert device.l1d_bytes_per_performance_core == 128 * 1024
    assert device.l1d_bytes_per_efficiency_core == 64 * 1024
    assert device.l2_bytes_per_performance_cluster == 16 * 1024 * 1024
    assert device.l2_bytes_per_efficiency_cluster == 4 * 1024 * 1024
    assert device.cache_line_bytes == 128
    assert device.performance_core_count == 8
    assert device.sm_count == 2


def test_amx_evidence_separates_a_measurement_from_a_reading():
    """The vendor's bandwidth was never measured here and the unit count is a
    reading of a scaling curve, so neither may pass as measured; the absent
    instruction peak is recorded as absent rather than guessed."""
    architecture = HARDWARE_SPECS.document("apple.amx")
    device = HARDWARE_SPECS.document("apple.m2_pro")

    assert device.fact("memory.unified.bandwidth").origin == "vendor"
    assert device.fact("compute.amx_unit_count").origin == "estimated"
    assert device.fact("throughput.amx.f32").origin == "measured"
    assert device.fact("throughput.neon.f32").origin == "measured"
    assert device.fact("geometry.neon_f32_outer_product").origin == "derived"
    assert not architecture.fact("throughput.f32_instruction_peak").available

    for document in (architecture, device):
        for fact in document.facts.values():
            assert fact.conditions


def test_the_two_storage_levels_stand_on_the_recorded_register_files():
    """The AMX level's per-operand budget is the recorded X/Y/Z geometry and
    nothing else; the cache level budgets no operand at all, which is what makes
    it hold anything and the filter vacuous for it."""
    architecture = AmxTarget().architecture
    assert AMX_REGISTERS.budget == (
        ("a_bytes", architecture.staging_bytes),
        ("b_bytes", architecture.staging_bytes),
        ("c_bytes", architecture.accumulator_bytes),
    )
    assert CORE_CACHE.budget == ()
    assert CORE_CACHE.holds({}) and CORE_CACHE.holds({"a_bytes": 10**12})


def test_a_core_tile_is_bounded_by_the_l1d_not_by_the_register_files():
    """A core-level tile's resident working set is bounded by the performance
    core's L1d; the register files bound one atom instance instead, which the
    storage filter enforces rather than a per-tile capacity."""
    target = AmxTarget()
    facts = TARGET_FACTS.project(target, TileStoreFacts, "core")
    assert facts.tile_capacity_bytes == target.device.l1d_bytes_per_performance_core
    assert target.architecture.accumulator_bytes < (
        target.device.l1d_bytes_per_performance_core
    )


def test_unmeasured_units_and_dtypes_have_no_throughput_entry():
    """FMA16 exists in the instruction set, but no f16 rate was measured on
    either unit, so asking for one raises instead of returning a guess -- as does
    naming a unit that has no recorded rate at all."""
    target = AmxTarget()
    assert target.architecture.supports_compute_dtype(DType.f16)
    assert target.device.throughput_for("amx", DType.f32) == 504_900_000_000
    assert target.device.throughput_for("neon", DType.f32) == 107_700_000_000
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        target.device.throughput_for("amx", DType.f16)
    with pytest.raises(ValueError, match="no measured compute-throughput"):
        target.device.throughput_for("sme", DType.f32)
