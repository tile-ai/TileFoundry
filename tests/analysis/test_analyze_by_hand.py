"""Exact analysis values for programs small enough to compute on paper."""

from __future__ import annotations

from dataclasses import replace

import isl

from tests.fixtures.placed.gemm_schedules import (
    WAVE_BK,
    WAVE_BM,
    WAVE_BN,
    WAVE_C,
    WAVE_G,
    WAVE_OTHER,
    GemmNaiveWave,
    GemmReuseAWave,
    GemmReuseBWave,
    GemmTile64,
    GemmTile128,
)
from tests.fixtures.placed.hand_checked import (
    BN,
    InvariantReuse,
    L2Occupancy,
    N,
    OneWave,
    OverlappingReads,
    PackedDtype,
    SlicedView,
    StoreOnly,
)
from tests.fixtures.placed.persistent_gemm_flat import PersistentGemmFlat
from tests.fixtures.placed.persistent_gemm_tiled import PersistentGemmTiled
from tilefoundry.analysis import analyze
from tilefoundry.analysis.footprint import ReachedAddresses, footprint_of, merged
from tilefoundry.analysis.report import report_data
from tilefoundry.ir.types import DType
from tilefoundry.target import CudaTarget


def _report(module) -> dict:
    result = analyze(
        module,
        module.entry_function(),
        analysis=("memory",),
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
    return memory["footprint"]["buffers"][name]["gmem"]["total"]


def _working_set_bytes(memory: dict) -> int:
    return sum(
        level["total"]
        for levels in memory["footprint"]["buffers"].values()
        for level in levels.values()
    )


def test_uncounted_boundary_marks_the_footprint_incomplete() -> None:
    buffer = InvariantReuse.entry_function().params[0]
    uncounted = ReachedAddresses(
        buffer=buffer,
        output_index=0,
        dtype=None,
        reached=None,
        exact=False,
    )

    footprint = footprint_of(
        merged((uncounted,)),
        memory_level="gmem",
        labels={id(buffer): "x"},
    )

    assert footprint.buffers == ()
    assert footprint.complete is False


def test_tuple_output_boundaries_add_instead_of_union() -> None:
    buffer = InvariantReuse.entry_function().params[0]
    addresses = isl.set("{ [i] : 0 <= i < 4 }")
    reached = tuple(
        ReachedAddresses(
            buffer=buffer,
            output_index=index,
            dtype=DType.bf16,
            reached=addresses,
            exact=True,
        )
        for index in range(2)
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
    assert footprint.complete is True


def test_invariant_reuse_matches_the_written_arithmetic() -> None:
    memory = _memory_record(InvariantReuse)
    traffic = memory["traffic"]["storage"]["gmem"]
    assert traffic["logical"] == {"read": 64, "write": 0}
    assert traffic["total"] == {"read": 192, "write": 0}
    assert traffic["per_unit"] == [{"read": 192, "write": 0}]
    assert _footprint_bytes(memory, "x") == 16
    assert N // BN == 3


def test_overlapping_reads_are_unioned_not_summed() -> None:
    memory = _memory_record(OverlappingReads)

    assert _footprint_bytes(memory, "x") == 12 * 2
    assert memory["footprint"]["complete"] is True


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


def test_one_wave_counts_only_the_resident_ctas() -> None:
    resident_data = _report(OneWave)
    resident = resident_data["function_records"]["memory"]
    wide_target = CudaTarget(
        replace(OneWave.target.device, sm_count=256),
        architecture=OneWave.target.architecture,
    )
    all_declared_data = _report(replace(OneWave, target=wide_target))
    all_declared = all_declared_data["function_records"]["memory"]

    assert resident_data["wave"] == {"counted": 132, "declared": 256}
    assert _footprint_bytes(resident, "x") == 132 * 4 * 2
    assert all_declared_data["wave"] == {"counted": 256, "declared": 256}
    assert _footprint_bytes(all_declared, "x") == 256 * 4 * 2


def test_l2_occupancy_matches_the_written_ratio() -> None:
    data = _report(L2Occupancy)
    memory = data["function_records"]["memory"]
    used = _working_set_bytes(memory)
    capacity = 1_048_576
    l2_errors = [error for error in memory["errors"] if error.startswith("l2 working set")]

    assert (used, capacity, used * 100 / capacity) == (1_572_864, 1_048_576, 150.0)
    assert l2_errors == [
        "l2 working set 1.50MB at the first iteration of a 1-unit wave "
        "exceeds capacity 1.00MB"
    ]
    assert memory["advisories"] == []


def test_persistent_tiled_holds_the_loop_at_its_start_expression() -> None:
    memory = _memory_record(PersistentGemmTiled)

    assert _footprint_bytes(memory, "a") == 4 * 64 * 32 * 2
    assert _footprint_bytes(memory, "b") == 4 * 32 * 64 * 2
    assert memory["footprint"]["complete"] is True


def test_persistent_flat_states_its_precision() -> None:
    """Derived a/b loads and the result store are the unbound operands.

    Their offsets depend on DimFloorDiv/DimMod of ``t``. Those widened
    boundaries make the Function footprint a lower bound.
    """
    data = _report(PersistentGemmFlat)
    memory = data["function_records"]["memory"]
    incomplete_calls = [
        call
        for call in data["calls"]
        if call["memory"]["footprint"]["complete"] is False
    ]
    incomplete_buffers = {
        name
        for call in incomplete_calls
        for name in call["memory"]["footprint"]["buffers"]
    }
    store = next(
        call
        for call in incomplete_calls
        if any(operand["arg"] == "result" for operand in call["memory"]["operands"])
        and any(operand["name"] == "out" for operand in call["memory"]["operands"])
    )

    assert memory["footprint"]["complete"] is False
    assert {"a", "b"} <= incomplete_buffers
    assert store["memory"]["footprint"]["complete"] is False
    assert _footprint_bytes(memory, "a") == 64 * 32 * 2
    assert _footprint_bytes(memory, "b") == 32 * 64 * 2
    assert _working_set_bytes(memory) > 64 * 64 * 4


def test_gemm_tile_shape_scales_traffic_and_working_set() -> None:
    tile64 = _memory_record(GemmTile64)
    tile128 = _memory_record(GemmTile128)

    assert tile64["traffic"]["storage"]["gmem"]["total"] != tile128["traffic"][
        "storage"
    ]["gmem"]["total"]
    assert _working_set_bytes(tile64) == 2 * 64 * 64 * 2
    assert _working_set_bytes(tile128) == 2 * 128 * 128 * 2
    assert _working_set_bytes(tile64) * 4 == _working_set_bytes(tile128)


def test_two_waves_share_traffic_but_differ_in_working_set() -> None:
    naive = _memory_record(GemmNaiveWave)
    reuse_a = _memory_record(GemmReuseAWave)
    reuse_b = _memory_record(GemmReuseBWave)
    total_reads = {
        memory["traffic"]["storage"]["gmem"]["total"]["read"]
        for memory in (naive, reuse_a, reuse_b)
    }

    assert total_reads == {3_244_032}
    assert _working_set_bytes(naive) == (WAVE_BM + WAVE_C * WAVE_BN) * WAVE_BK * 2
    assert _working_set_bytes(reuse_a) != _working_set_bytes(reuse_b)


def test_deepgemm_waves_match_both_scheduler_formulas() -> None:
    reuse_a = _memory_record(GemmReuseAWave)
    reuse_b = _memory_record(GemmReuseBWave)
    expected_a = (WAVE_OTHER * WAVE_BM + WAVE_G * WAVE_BN) * WAVE_BK * 2
    expected_b = (WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN) * WAVE_BK * 2

    assert _working_set_bytes(reuse_a) == expected_a == 143_360
    assert _working_set_bytes(reuse_b) == expected_b == 139_264


def test_analyzer_agrees_with_deepgemm_min_choice() -> None:
    analyzed = {
        "reuse_a": _working_set_bytes(_memory_record(GemmReuseAWave)),
        "reuse_b": _working_set_bytes(_memory_record(GemmReuseBWave)),
    }
    official = {
        "reuse_a": WAVE_G * WAVE_BN + WAVE_OTHER * WAVE_BM,
        "reuse_b": WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN,
    }

    assert min(analyzed, key=analyzed.get) == min(official, key=official.get) == "reuse_b"
