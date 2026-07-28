"""The AMX target's own facts: its topology levels, its atom candidate
enumeration and the recorded origin of every fact it stands on.

The device facts are the ones actually measured on the host this target
describes, so the assertions below name the exact values rather than a range --
in particular the performance-core cache sizes, which the unqualified
``hw.l1dcachesize`` / ``hw.l2cachesize`` sysctls under-report by naming the
efficiency core instead. A duration derived from a rate nobody measured is the
failure this file exists to prevent, and it is silent: the plan still comes out,
priced against a machine that does not exist.
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.schedule.pipeline.facts import PipelineFacts, PipelineFactsQuery
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


def test_amx_target_reports_and_validates_its_own_topology_levels():
    """The AMX levels are the core a tile stream runs on and the AMX unit inside
    it that issues one atom. The core limit is the measured performance-core
    count; the unit limit comes from the architecture.

    Asking about a level the target does not have raises, and the message lists
    the levels it does -- for the limit lookup and for the declared-topology
    validation alike, since a program naming a foreign level is the same mistake
    either way. A declared extent is validated against its level's own limit, and
    an AMX core count is always static: there is no launch shape to defer it to,
    so a symbolic extent is refused rather than accepted and counted later.
    """
    target = AmxTarget()
    assert target.topology_levels == ("core", "amx")
    assert target.topology_limit("core") == 8
    assert target.topology_limit("amx") == 1
    assert resolve_target("amx").topology_levels == target.topology_levels

    with pytest.raises(ValueError) as limit_error:
        target.topology_limit("cta")
    with pytest.raises(ValueError) as topology_error:
        target.validate_program_topology(Topology("cta", 4))
    for error in (limit_error, topology_error):
        message = str(error.value)
        assert "unsupported topology level 'cta'" in message
        assert "('core', 'amx')" in message

    target.validate_program_topology(Topology("core", 8))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 8"):
        target.validate_program_topology(Topology("core", 9))
    with pytest.raises(ValueError, match="must be positive"):
        target.validate_program_topology(Topology("core", 0))
    with pytest.raises(ValueError, match="requires a positive static integer"):
        target.validate_program_topology(Topology("core", DimVar("cores", 1, 8)))


# ---------------------------------------------------------------------------
# AC-4-2
# ---------------------------------------------------------------------------


def test_both_catalogue_atoms_are_priced_at_their_own_measured_rates():
    """AC-4-2: the two units of one core, each listed with its real numbers.

    A gemm whose M=16, N=16 divide the AMX atom's 16x16 shape, whose K divides its
    own extent of one, and whose three operands fit the X/Y/Z register files gets
    both candidates. The AMX one is a 16x16 f32 outer product: a 16-element A
    operand in X, a 16-element B operand in Y, and the accumulator tile it lands
    in, in Z. The NEON one is a 4x4 f32 outer product on the core's SIMD pipes,
    its operands streamed through cache rather than held in registers.

    Every field is checked against the atom's own numbers, and both durations
    against the measured rate of the unit that runs them -- 2*M*N*K flops over the
    unit's own throughput, operand bytes over the unified-memory bandwidth. At this
    granularity both are memory-bound, which is the arithmetic saying so rather
    than a claim about the hardware.
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
    """AC-4-2: three independent reasons an atom is not a candidate.

    Storage is what separates the two units, and it is the recorded X/Y/Z geometry
    that decides it: an untiled 64x128 f32 A operand is 32 KiB against 512 B of X,
    so only the cache-streaming NEON atom survives -- which is why a whole-tensor
    statement lands on SIMD. The cache level budgets no operand at all, so its
    filter is vacuous by construction rather than by omission.

    Shape is decided per atom, not per catalogue: M=8 fits the register files
    whole, so nothing about storage rules the AMX atom out -- its own 16-row
    granularity does, while NEON's 4 divides 8. An extent of 18 is a whole
    multiple of neither, and a bf16 gemm matches neither atom's dtype. Each is an
    empty list rather than an error, so "no atom applies" stays distinguishable
    from "this cannot be asked".
    """
    architecture = AmxTarget().architecture
    assert AMX_REGISTERS.budget == (
        ("a_bytes", architecture.staging_bytes),
        ("b_bytes", architecture.staging_bytes),
        ("c_bytes", architecture.accumulator_bytes),
    )
    assert CORE_CACHE.budget == ()
    assert CORE_CACHE.holds({}) and CORE_CACHE.holds({"a_bytes": 10**12})

    whole_tensor = candidate_atoms(
        f32_gemm.entry_function().body, f32_gemm.resolve_target()
    )
    assert [fact.atom.op.name for fact in whole_tensor] == ["NEON_FMLA_4x4x1_F32"]
    assert not AMX_REGISTERS.holds(
        {"a_bytes": 64 * 128 * 4, "b_bytes": 512, "c_bytes": 512}
    )
    assert CORE_CACHE.holds({"a_bytes": 64 * 128 * 4, "b_bytes": 10**9, "c_bytes": 10**9})

    coarse = candidate_atoms(
        coarse_m_f32_gemm.entry_function().body, coarse_m_f32_gemm.resolve_target()
    )
    assert [fact.atom.op.name for fact in coarse] == ["NEON_FMLA_4x4x1_F32"]
    assert AMX_REGISTERS.holds(
        {"a_bytes": 8 * 8 * 4, "b_bytes": 8 * 16 * 4, "c_bytes": 8 * 16 * 4}
    )

    for indivisible in (odd_m_f32_gemm, odd_n_f32_gemm):
        assert candidate_atoms(
            indivisible.entry_function().body, indivisible.resolve_target()
        ) == []
    assert candidate_atoms(bf16_gemm.entry_function().body, bf16_gemm.resolve_target()) == []


# ---------------------------------------------------------------------------
# AC-4-3
# ---------------------------------------------------------------------------

def test_amx_values_stand_on_the_installed_documents_and_say_how_they_were_got():
    """AC-4-3: the architecture carries the ISA geometry, the device the per-part
    resources, each built from its own installed document -- so a number moved to
    the other side would be claimed of the wrong thing.

    And each fact records how it was obtained. The vendor's bandwidth was never
    measured here and the unit count is a reading of a scaling curve, so neither
    may pass as measured; the absent instruction peak is recorded as absent rather
    than guessed. Every fact states its conditions, because a throughput without
    them is a number without a meaning.
    """
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

    architecture_document = HARDWARE_SPECS.document("apple.amx")
    device_document = HARDWARE_SPECS.document("apple.m2_pro")
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
    """A core-level tile's resident working set is bounded by the performance
    core's L1d; the register files bound one atom instance instead, which the
    storage filter enforces rather than a per-tile capacity."""
    target = AmxTarget()
    facts = TARGET_FACTS.project(
        target, PipelineFacts, PipelineFactsQuery(topology="core", statements=())
    )
    assert facts.tile_capacity_bytes == target.device.l1d_bytes_per_performance_core
    assert facts.tile_capacity_scope == "core"
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
