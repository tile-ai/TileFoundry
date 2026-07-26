"""The AMX target's own facts: its topology levels, its atom candidate
enumeration and the provenance of every device fact it stands on.

The device facts are the ones actually measured on the host this target
describes, so the assertions below name the exact values rather than a range --
in particular the performance-core cache sizes, which the unqualified
``hw.l1dcachesize`` / ``hw.l2cachesize`` sysctls under-report by naming the
efficiency core instead.
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.analysis import Analysis, AtomFact
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import Schedule
from tilefoundry.target import AmxTarget, resolve_target
from tilefoundry.target.amx.atoms import AMX_FMA32_16x16x1_F32, AmxAtom, candidate_atoms
from tilefoundry.target.hardware import load_apple_m2_pro_amx, load_hardware_spec


@func(target="amx")
def f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def odd_m_f32_gemm(
    x: Tensor[(24, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(24, 64), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def odd_n_f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 24), "f32"],
) -> Tensor[(64, 24), "f32"]:
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
    """The core stage binds one Analysis and one Schedule service; a stage the
    target does not serve raises."""
    target = AmxTarget()
    assert target.service(Analysis, "core").stage == "core"
    assert target.service(Schedule, "core").stage == "core"
    with pytest.raises(ValueError, match="expected exactly one service"):
        target.service(Analysis, "cta")


# ---------------------------------------------------------------------------
# AC-4-2
# ---------------------------------------------------------------------------


def test_a_divisible_f32_gemm_lists_the_amx_outer_product_atom():
    """AC-4-2: M=64, N=64 both divide the atom's 16x16 shape and K divides its
    own extent of one, so the f32 outer product is a candidate. Every field is
    checked against the atom's real numbers, not a placeholder."""
    facts = candidate_atoms(f32_gemm.body, f32_gemm.target)
    print("\n=== candidate AtomFacts (f32 gemm, M=64 N=64 K=128) ===")
    for fact in facts:
        print(fact)

    assert len(facts) == 1
    fact = facts[0]
    assert isinstance(fact, AtomFact)
    assert fact.shape == (16, 16, 1)
    assert fact.dtype == (DType.f32, DType.f32, DType.f32)
    assert isinstance(fact.atom, AmxAtom)
    assert fact.atom.op is AMX_FMA32_16x16x1_F32

    # One 16x16 f32 outer product: a 16-element X operand, a 16-element Y
    # operand and the 16x16 f32 accumulator tile it lands in.
    assert fact.storage == {
        "x_bytes": 64,
        "y_bytes": 64,
        "z_bytes": 1024,
        "register_bytes": 1152,
    }
    assert fact.resource == {"amx": 1}
    assert fact.is_async is False

    # 2*16*16*1 flops over the measured per-unit f32 rate; 1152 bytes over the
    # unified-memory bandwidth. The atom is memory-bound at this granularity.
    assert fact.compute_duration == pytest.approx(512e9 / 504_900_000_000)
    assert fact.duration == pytest.approx(1152e9 / 200_000_000_000)
    assert fact.duration > fact.compute_duration


@pytest.mark.parametrize("fn", [odd_m_f32_gemm, odd_n_f32_gemm])
def test_an_indivisible_extent_lists_no_amx_candidate(fn):
    """AC-4-2: an extent of 24 is not a whole multiple of the atom's 16, on
    either M or N -- hard-filtered out, an empty list rather than an error."""
    assert candidate_atoms(fn.body, fn.target) == []


def test_a_non_f32_gemm_lists_no_amx_candidate():
    """The modelled catalogue is FMA32 only, so a bf16 gemm's operand dtypes do
    not match any registered atom."""
    assert candidate_atoms(bf16_gemm.body, bf16_gemm.target) == []


def test_candidate_atoms_defaults_and_resolves_a_backend_name():
    """``target=None`` defaults to the AMX target and a ``target=`` string is
    resolved, matching ``@func(target="amx")``'s own surface."""
    explicit = candidate_atoms(f32_gemm.body, AmxTarget())
    assert candidate_atoms(f32_gemm.body) == explicit
    assert candidate_atoms(f32_gemm.body, "amx") == explicit


def test_a_non_amx_target_raises():
    """The per-atom device facts this bridge reads (per-unit AMX throughput,
    unified-memory bandwidth) exist on the AMX device only."""
    with pytest.raises(NotImplementedError, match="only AmxTarget is supported"):
        candidate_atoms(f32_gemm.body, target="cuda")


# ---------------------------------------------------------------------------
# AC-4-3
# ---------------------------------------------------------------------------

_PROVENANCE = {"measured", "derived", "direct", "vendor-spec", "estimated", "unavailable"}


def test_every_amx_device_fact_carries_a_provenance_entry():
    """AC-4-3: every fact in the AMX hardware specification records where its
    value came from, with a condition explaining what it holds under."""
    spec = load_apple_m2_pro_amx()
    assert spec["target"]["name"] == "apple_m2_pro_amx"

    for name, fact in spec["facts"].items():
        assert set(fact) == {"value", "unit", "provenance", "conditions", "source"}, name
        assert fact["provenance"] in _PROVENANCE, (name, fact["provenance"])
        assert fact["conditions"], name
    assert load_hardware_spec(AmxTarget()) == spec


def test_unmeasured_amx_facts_are_marked_as_such():
    """AC-4-3: the vendor's bandwidth figure was never measured here and the
    AMX unit count is a reading of a scaling curve, so neither is allowed to
    pass as measured; the absent instruction peak is recorded as absent."""
    facts = load_apple_m2_pro_amx()["facts"]

    assert facts["unified_memory_bandwidth"]["provenance"] == "vendor-spec"
    assert facts["amx_unit_count"]["provenance"] == "estimated"
    assert facts["amx_f32_instruction_peak"]["provenance"] == "unavailable"
    assert facts["amx_f32_unit_throughput"]["provenance"] == "measured"
    assert facts["amx_z_register_file"]["value"] == 4096


def test_device_values_match_the_facts_they_are_recorded_from():
    """The device stands on the recorded facts: the performance-core cache
    sizes, the AMX register files and the two parallelism counts."""
    device = AmxTarget().device
    facts = load_apple_m2_pro_amx()["facts"]

    assert device.l1d_bytes_per_performance_core == facts["l1d_per_performance_core"]["value"] * 1024
    assert device.l2_bytes_per_performance_cluster == (
        facts["l2_per_performance_cluster"]["value"] * 1024 * 1024
    )
    assert device.amx_staging_bytes == facts["amx_x_register_file"]["value"]
    assert device.amx_accumulator_bytes == facts["amx_z_register_file"]["value"]
    assert device.performance_core_count == facts["performance_core_count"]["value"]
    assert device.sm_count == facts["amx_unit_count"]["value"]
    assert device.cache_line_bytes == facts["cache_line"]["value"]

    # The performance core's caches, not the efficiency core's 64 KiB / 4 MiB.
    assert device.l1d_bytes_per_performance_core == 128 * 1024
    assert device.l1d_bytes_per_efficiency_core == 64 * 1024
    assert device.l2_bytes_per_efficiency_cluster == 4 * 1024 * 1024


def test_the_tile_capacity_fact_is_the_accumulator_not_a_cache():
    """The capacity a tile's footprint is charged against is the Z accumulator
    file, which is what a resident accumulation is bounded by -- an order of
    magnitude below the performance core's L1d."""
    device = AmxTarget().device
    assert device.l1_capacity_bytes == 4096
    assert device.l1_capacity_bytes == device.amx_accumulator_bytes
    assert device.l1_capacity_bytes < device.l1d_bytes_per_performance_core
    assert device.l2_bandwidth_bytes_per_second == 200_000_000_000


def test_unmeasured_compute_dtypes_have_no_throughput_entry():
    """FMA16 exists in the instruction set, but no f16 rate was measured, so
    asking for one raises instead of returning a guess."""
    target = AmxTarget()
    assert target.architecture.supports_compute_dtype(DType.f16)
    assert target.device.throughput_for(DType.f32) == 504_900_000_000
    with pytest.raises(ValueError, match="no measured AMX compute-throughput"):
        target.device.throughput_for(DType.f16)
