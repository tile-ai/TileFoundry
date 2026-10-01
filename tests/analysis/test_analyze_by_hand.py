"""Exact analysis values for programs small enough to compute on paper."""

from __future__ import annotations

from dataclasses import replace

import isl
import pytest

from tests.fixtures.placed.gemm_schedules import (
    WAVE_BK,
    WAVE_BM,
    WAVE_BN,
    WAVE_C,
    WAVE_G,
    WAVE_OTHER,
    Gemm_MK_NN64x128x32_w1x132,
    Gemm_MNK_NN64,
    Gemm_MNK_NN64x128x32_w11x12,
    Gemm_MNK_NN64x128x32_w12x11,
    Gemm_MNK_NN128,
    Gemm_MNK_NN128x128x64_w12x11_k4096,
    Gemm_MNK_NN128x128x64_w12x11_k16384,
    Gemm_MNK_NT128x128x64_w17x8,
)
from tests.fixtures.placed.hand_checked import (
    BN,
    CapacityExceeded,
    InvariantReuse,
    N,
    OverlappingReads,
    PackedDtype,
    SiblingLoopReuse,
    SlicedView,
    StoreOnly,
    TruncatedWaveReuse,
    WaveTruncation,
)
from tests.fixtures.placed.persistent_gemm_flat import PersistentGemmFlat
from tests.fixtures.placed.persistent_gemm_tiled import PersistentGemmTiled
from tilefoundry.analysis import (
    AnalysisPrecision,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    analyze,
)
from tilefoundry.analysis import footprint as footprint_analysis
from tilefoundry.analysis import memory as memory_analysis
from tilefoundry.analysis.compute_cost import local_duration_ns
from tilefoundry.analysis.footprint import ReachedAddresses, footprint_of, merged
from tilefoundry.analysis.iteration_scope import build_scopes, walk_scopes
from tilefoundry.analysis.report import report_data
from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import DType
from tilefoundry.target import CudaTarget, PerformanceServiceFacts, ThroughputFacts


def _report(module, *, analysis=("memory",)) -> dict:
    result = analyze(
        module,
        module.entry_function(),
        analysis=analysis,
    )
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )
    return data


def _memory_record(module) -> dict:
    return _report(module)["function_records"]["memory"]


def _footprint_bytes(memory: dict, name: str) -> int:
    return memory["footprint"]["buffers"][name]["gmem"]["logical"]


def _working_set_bytes(memory: dict) -> int:
    return sum(
        level["logical"]
        for levels in memory["footprint"]["buffers"].values()
        for level in levels.values()
    )


def _reuse_conclusions(memory: dict) -> list[dict]:
    fields = ("buffer", "time", "space", "holds_bytes", "reuse_bytes", "fits")
    return [{field: row[field] for field in fields} for row in memory["reuse_windows"]]


@pytest.mark.parametrize("counted_precision", tuple(AnalysisPrecision))
@pytest.mark.parametrize(
    "missing_precision", (AnalysisPrecision.LOWER_BOUND, AnalysisPrecision.UNKNOWN)
)
def test_uncounted_boundary_marks_the_footprint_incomplete(
    counted_precision: AnalysisPrecision,
    missing_precision: AnalysisPrecision,
) -> None:
    buffer = InvariantReuse.entry_function().params[0]
    uncounted = ReachedAddresses(
        buffer=buffer,
        output_index=0,
        dtype=None,
        reached=None,
        precision=missing_precision,
    )
    counted = ReachedAddresses(
        buffer, 0, DType.bf16, isl.set("{ [i] : 0 <= i < 4 }"), counted_precision
    )

    footprint = footprint_of(
        merged((uncounted, counted)),
        memory_level="gmem",
        labels={id(buffer): "x"},
    )

    assert dict(footprint.buffers)["x"].of("gmem").total == 8
    expected = (
        missing_precision
        if counted_precision in (AnalysisPrecision.EXACT, missing_precision)
        else AnalysisPrecision.UNKNOWN
    )
    assert footprint.precision is expected
    assert (
        footprint_of((uncounted,), memory_level="gmem", labels={id(buffer): "x"}).precision
        is missing_precision
    )


@pytest.mark.parametrize("left", tuple(AnalysisPrecision))
@pytest.mark.parametrize("right", tuple(AnalysisPrecision))
def test_tuple_output_boundaries_add_instead_of_union(
    left: AnalysisPrecision, right: AnalysisPrecision,
) -> None:
    buffer = InvariantReuse.entry_function().params[0]
    addresses = isl.set("{ [i] : 0 <= i < 4 }")
    reached = tuple(
        ReachedAddresses(
            buffer=buffer,
            output_index=index,
            dtype=DType.bf16,
            reached=addresses,
            precision=precision,
        )
        for index, precision in enumerate((left, right))
    )

    distinct = merged(reached)
    footprint = footprint_of(
        distinct,
        memory_level="gmem",
        labels={id(buffer): "x"},
    )
    by_name = dict(footprint.buffers)
    gmem = by_name["x"].of("gmem")

    assert len(distinct) == 2
    assert gmem is not None and gmem.total == 2 * 4 * 2
    expected = (
        right if left is AnalysisPrecision.EXACT
        else left if right is AnalysisPrecision.EXACT or left is right
        else AnalysisPrecision.UNKNOWN
    )
    assert footprint.precision is expected


def test_invariant_reuse_matches_the_written_arithmetic() -> None:
    result = analyze(
        InvariantReuse,
        InvariantReuse.entry_function(),
        analysis=("compute-cost", "memory", "roofline", "performance"),
    )
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )
    memory = data["function_records"]["memory"]
    traffic = memory["traffic"]["storage"]["gmem"]
    assert traffic["logical"] == {"read": 64, "write": 0}
    assert traffic["total"] == {"read": 768, "write": 0}
    assert traffic["per_unit"] == [{"read": 192, "write": 0}]
    assert _footprint_bytes(memory, "x") == 16
    footprint_total = memory["footprint"]["buffers"]["x"]["gmem"]["total"]
    assert footprint_total == 192
    assert data["function_records"]["roofline"]["memory_ns"] == -(
        -(footprint_total * 1_000_000_000) // 4_800_000_000_000
    )
    assert N // BN == 3
    assert data["function_records"]["compute-cost"]["flops"]["bf16"] == {
        "logical": 32, "total": 384, "per_unit": [96],
    }
    scopes = tuple(walk_scopes(build_scopes(result.module, result.function)))
    loaded = next(
        call
        for scope in scopes
        for call, _ in scope.accesses["narrow"].values()
        if isinstance(call.target, Reshard)
    )
    n_scope = next(
        scope for scope in scopes
        if isinstance(scope.owner, LoopRegion) and scope.owner.induction_var.name == "n"
    )
    assert not n_scope.is_variant(loaded)
    target = result.module.resolve_target()
    duration = local_duration_ns(
        get_metadata(loaded, ComputeCostMetadata),
        target.get_facts(ThroughputFacts),
        target.get_facts(PerformanceServiceFacts, result.topology_level),
        moved=get_metadata(loaded, MemoryMetadata),
        level=result.topology_level,
    )
    timeline = get_metadata(loaded, PerformanceMetadata).timeline
    assert (timeline.end_ns - timeline.start_ns) * timeline.trips == duration * 12


def test_overlapping_reads_are_unioned_not_summed() -> None:
    memory = _memory_record(OverlappingReads)

    assert _footprint_bytes(memory, "x") == 12 * 2
    assert memory["footprint"]["precision"] == "exact"


def test_sliced_view_counts_against_the_final_source() -> None:
    memory = _memory_record(SlicedView)

    assert _footprint_bytes(memory, "x") == 8 * 4
    assert set(memory["footprint"]["buffers"]) == {"x"}


def test_store_only_still_occupies_the_cache() -> None:
    memory = _memory_record(StoreOnly)
    traffic = memory["traffic"]["storage"]["gmem"]

    assert _working_set_bytes(memory) == 8 * 2
    assert traffic["total"] == {"read": 0, "write": 16}


def test_packed_dtype_rounds_up_to_whole_bytes() -> None:
    memory = _memory_record(PackedDtype)

    assert _footprint_bytes(memory, "x") == 5


def test_wave_truncation_counts_only_the_resident_ctas() -> None:
    resident_data = _report(WaveTruncation, analysis=("memory", "roofline"))
    resident = resident_data["function_records"]["memory"]
    wide_target = CudaTarget(
        replace(WaveTruncation.target.device, sm_count=256),
        architecture=WaveTruncation.target.architecture,
    )
    all_declared_data = _report(
        replace(WaveTruncation, target=wide_target), analysis=("memory", "roofline")
    )
    all_declared = all_declared_data["function_records"]["memory"]

    assert resident_data["wave"] == {"counted": 132, "declared": 256}
    assert _footprint_bytes(resident, "x") == 132 * 4 * 2
    assert all_declared_data["wave"] == {"counted": 256, "declared": 256}
    assert resident["traffic"]["storage"]["gmem"]["total"] == {"read": 2048, "write": 0}
    assert all_declared["traffic"]["storage"]["gmem"]["total"] == {"read": 2048, "write": 0}
    assert _footprint_bytes(all_declared, "x") == 256 * 4 * 2
    assert resident["footprint"]["buffers"]["x"]["gmem"]["total"] == 2112
    assert all_declared["footprint"]["buffers"]["x"]["gmem"]["total"] == 2048
    assert resident_data["function_records"]["roofline"]["memory_ns"] == -(
        -(2112 * 1_000_000_000) // 4_800_000_000_000
    )


def test_truncated_wave_keeps_reuse_from_participating_boundaries() -> None:
    data = _report(TruncatedWaveReuse, analysis=("memory", "roofline"))
    memory = data["function_records"]["memory"]

    assert data["wave"] == {"counted": 132, "declared": 256}
    assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 8192, "write": 0}
    assert _working_set_bytes(memory) == 32
    assert sum(
        levels["gmem"]["total"] for levels in memory["footprint"]["buffers"].values()
    ) == 64
    assert data["function_records"]["roofline"]["memory_ns"] == -(
        -(64 * 1_000_000_000) // 4_800_000_000_000
    )
    assert _reuse_conclusions(memory) == [
        {
            "buffer": "<value 0>",
            "time": "",
            "space": "cta.i",
            "holds_bytes": 32,
            "reuse_bytes": 4_192,
            "fits": True,
        }
    ]


def test_time_window_excludes_a_buffer_in_a_sibling_loop() -> None:
    memory = _memory_record(SiblingLoopReuse)

    assert set(memory["footprint"]["buffers"]) == {"x", "y"}
    assert _reuse_conclusions(memory) == [
        {
            "buffer": "x",
            "time": "n",
            "space": "",
            "holds_bytes": 16,
            "reuse_bytes": 32,
            "fits": True,
        }
    ]


def test_deepgemm_wave_reuse_conclusions_match_the_reviewed_report() -> None:
    memory = _memory_record(Gemm_MNK_NT128x128x64_w17x8)

    assert _reuse_conclusions(memory) == [
        {
            "buffer": "b",
            "time": "",
            "space": "cta.x",
            "holds_bytes": 409_600,
            "reuse_bytes": 2_097_152,
            "fits": True,
        },
        {
            "buffer": "a",
            "time": "",
            "space": "cta.y",
            "holds_bytes": 409_600,
            "reuse_bytes": 1_949_696,
            "fits": True,
        },
    ]


def test_persistent_gemm_fits_reuse_conclusions_match_the_reviewed_report() -> None:
    memory = _memory_record(Gemm_MNK_NN128x128x64_w12x11_k4096)

    assert _reuse_conclusions(memory) == [
        {
            "buffer": "b",
            "time": "mi",
            "space": "cta.x",
            "holds_bytes": 46_137_344,
            "reuse_bytes": 1_174_405_120,
            "fits": True,
        },
        {
            "buffer": "a",
            "time": "ni",
            "space": "cta.y",
            "holds_bytes": 24_117_248,
            "reuse_bytes": 402_653_184,
            "fits": True,
        },
    ]


@pytest.mark.parametrize("precision", tuple(AnalysisPrecision))
def test_persistent_gemm_over_reuse_conclusions_match_the_reviewed_report(
    precision: AnalysisPrecision, monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = footprint_analysis.footprint_of
    monkeypatch.setattr(
        footprint_analysis, "footprint_of",
        lambda *args, **kwargs: replace(count(*args, **kwargs), precision=precision),
    )
    memory = _memory_record(Gemm_MNK_NN128x128x64_w12x11_k16384)

    assert _reuse_conclusions(memory) == [
        {
            "buffer": "b",
            "time": "mi",
            "space": "cta.x",
            "holds_bytes": 184_549_376,
            "reuse_bytes": 4_697_620_480,
            "fits": False,
        },
        {
            "buffer": "a",
            "time": "ni",
            "space": "cta.y",
            "holds_bytes": 96_468_992,
            "reuse_bytes": 1_610_612_736,
            "fits": False,
        },
    ]
    category = "errors" if precision in (
        AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND
    ) else "advisories"
    assert all(row["precision"] == precision.value for row in memory["reuse_windows"])
    assert memory[category] == [
        "l2 reuse window mi holds 176.00MB at a 132-unit wave, exceeding capacity 47.68MB",
        "l2 reuse window ni holds 92.00MB at a 132-unit wave, exceeding capacity 47.68MB",
    ]
    assert memory["advisories" if category == "errors" else "errors"] == []


@pytest.mark.parametrize("precision", tuple(AnalysisPrecision))
def test_capacity_exceeded_matches_the_written_ratio(
    precision: AnalysisPrecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = memory_analysis.footprint_of
    monkeypatch.setattr(
        memory_analysis,
        "footprint_of",
        lambda *args, **kwargs: replace(count(*args, **kwargs), precision=precision),
    )
    data = _report(CapacityExceeded)
    memory = data["function_records"]["memory"]
    used = _working_set_bytes(memory)
    capacity = 1_048_576
    category = (
        "errors"
        if precision in (AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND)
        else "advisories"
    )

    assert (used, capacity, used * 100 / capacity) == (1_572_864, 1_048_576, 150.0)
    diagnosis = [
        "l2 working set 1.50MB at the first iteration of a 1-unit wave exceeds capacity 1.00MB"
    ]
    assert memory["errors"] == [
        "rmem placement peak 1.50MB exceeds capacity 256.00KB"
    ] + (diagnosis if category == "errors" else [])
    assert memory["advisories"] == (diagnosis if category == "advisories" else [])


def test_persistent_tiled_holds_the_loop_at_its_start_expression() -> None:
    memory = _memory_record(PersistentGemmTiled)

    assert _footprint_bytes(memory, "a") == 12 * 64 * 32 * 2
    assert _footprint_bytes(memory, "b") == 11 * 32 * 64 * 2
    assert memory["footprint"]["precision"] == "exact"


def test_persistent_flat_states_its_precision() -> None:
    """Floor/mod offsets retain the exact first wave's loads and written tiles.

    Affine rendering handles ``t // 66`` and ``t % 66`` without widening.
    The first wave's ``t = 0..131`` reaches two A row blocks, all 66 B
    column blocks, and 132 distinct output tiles.
    """
    data = _report(PersistentGemmFlat)
    memory = data["function_records"]["memory"]
    store = next(
        call
        for call in data["calls"]
        if any(operand["arg"] == "result" for operand in call["memory"]["operands"])
        and any(operand["name"] == "out" for operand in call["memory"]["operands"])
    )

    assert memory["footprint"]["precision"] == "exact"
    assert store["memory"]["footprint"]["precision"] == "exact"
    assert _footprint_bytes(memory, "a") == 2 * 64 * 32 * 2
    assert _footprint_bytes(memory, "b") == 66 * 32 * 64 * 2
    assert _working_set_bytes(memory) == 2_473_984
    assert sorted(
        levels["gmem"]["logical"] for levels in store["memory"]["footprint"]["buffers"].values()
    ) == [64 * 64 * 4, 132 * 64 * 64 * 4]


def test_tile_area_scales_traffic_and_working_set() -> None:
    tile64 = _memory_record(Gemm_MNK_NN64)
    tile128 = _memory_record(Gemm_MNK_NN128)

    assert (
        tile64["traffic"]["storage"]["gmem"]["total"]
        != tile128["traffic"]["storage"]["gmem"]["total"]
    )
    assert _working_set_bytes(tile64) == 2 * 64 * 64 * 2
    assert _working_set_bytes(tile128) == 2 * 128 * 128 * 2
    assert _working_set_bytes(tile64) * 4 == _working_set_bytes(tile128)


def test_two_waves_share_traffic_but_differ_in_working_set() -> None:
    naive = _memory_record(Gemm_MK_NN64x128x32_w1x132)
    reuse_a = _memory_record(Gemm_MNK_NN64x128x32_w11x12)
    reuse_b = _memory_record(Gemm_MNK_NN64x128x32_w12x11)
    total_reads = {
        memory["traffic"]["storage"]["gmem"]["total"]["read"]
        for memory in (naive, reuse_a, reuse_b)
    }

    assert total_reads == {428_212_224}
    assert _working_set_bytes(naive) == (WAVE_BM + WAVE_C * WAVE_BN) * WAVE_BK * 2
    assert _working_set_bytes(reuse_a) != _working_set_bytes(reuse_b)


def test_deepgemm_waves_match_both_scheduler_formulas() -> None:
    reuse_a = _memory_record(Gemm_MNK_NN64x128x32_w11x12)
    reuse_b = _memory_record(Gemm_MNK_NN64x128x32_w12x11)
    expected_a = (WAVE_OTHER * WAVE_BM + WAVE_G * WAVE_BN) * WAVE_BK * 2
    expected_b = (WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN) * WAVE_BK * 2

    assert _working_set_bytes(reuse_a) == expected_a == 143_360
    assert _working_set_bytes(reuse_b) == expected_b == 139_264


def test_analyzer_agrees_with_deepgemm_min_choice() -> None:
    analyzed = {
        "reuse_a": _working_set_bytes(_memory_record(Gemm_MNK_NN64x128x32_w11x12)),
        "reuse_b": _working_set_bytes(_memory_record(Gemm_MNK_NN64x128x32_w12x11)),
    }
    official = {
        "reuse_a": WAVE_G * WAVE_BN + WAVE_OTHER * WAVE_BM,
        "reuse_b": WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN,
    }

    assert min(analyzed, key=analyzed.get) == min(official, key=official.get) == "reuse_b"
