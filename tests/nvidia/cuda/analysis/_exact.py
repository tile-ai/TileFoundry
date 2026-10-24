"""Exact arithmetic checks attached to matching analysis corpus cases."""

from __future__ import annotations

from tests.nvidia.cuda.analysis._contracts import (
    _footprint_bytes,
    _reuse_conclusions,
    _working_set_bytes,
)
from tests.nvidia.cuda.fixtures.placed import persistent_gemm_tiled
from tests.nvidia.cuda.fixtures.placed.data_started_window import C, D, DataStart, MeshStart, W
from tests.nvidia.cuda.fixtures.placed.gemm_schedules import (
    WAVE_BK,
    WAVE_BM,
    WAVE_BN,
    WAVE_C,
    WAVE_G,
    WAVE_OTHER,
)
from tests.nvidia.cuda.fixtures.placed.hand_checked import (
    BN,
    N,
)
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
)
from tilefoundry.analysis.compute_cost import local_duration_ns
from tilefoundry.analysis.iteration_scope import build_scopes
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import PerformanceServiceFacts, ThroughputFacts


def _first_duration(result, kind) -> int:
    call = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, kind)
    )
    timeline = get_metadata(call, PerformanceMetadata).timeline
    return timeline.end_ns - timeline.start_ns


def _overlapping_reads_are_unioned_not_summed(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 12 * 2
        assert memory["footprint"]["precision"] == "exact"

    problems.run("_overlapping_reads_are_unioned_not_summed", check, state.report)


def _sliced_view_counts_against_the_final_source(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 8 * 4
        assert set(memory["footprint"]["buffers"]) == {"x"}

    problems.run("_sliced_view_counts_against_the_final_source", check, state.report)


def _store_only_still_occupies_the_cache(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        traffic = memory["traffic"]["storage"]["gmem"]

        assert _working_set_bytes(memory) == 8 * 2
        assert traffic["total"] == {"read": 0, "write": 16}

    problems.run("_store_only_still_occupies_the_cache", check, state.report)


def _packed_dtype_rounds_up_to_whole_bytes(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 5

    problems.run("_packed_dtype_rounds_up_to_whole_bytes", check, state.report)


def _truncated_wave_keeps_reuse_from_participating_boundaries(state, problems):
    def check(report):
        data = report
        memory = data["functions"][0]["function_records"]["memory"]

        assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 8192, "write": 0}
        assert _working_set_bytes(memory) == 32
        assert (
            sum(levels["gmem"]["total"] for levels in memory["footprint"]["buffers"].values()) == 64
        )
        assert data["functions"][0]["function_records"]["roofline"]["memory_ns"] == -(
            -(64 * 1_000_000_000) // 4_800_000_000_000
        )
        assert _reuse_conclusions(memory) == [
            {
                "buffer": "view::viewed:10",
                "time": "",
                "space": "cta.i",
                "holds_bytes": 32,
                "reuse_bytes": 4_192,
                "fits": True,
            }
        ]

    problems.run("_truncated_wave_keeps_reuse_from_participating_boundaries", check, state.report)


def _time_window_excludes_a_buffer_in_a_sibling_loop(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

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

    problems.run("_time_window_excludes_a_buffer_in_a_sibling_loop", check, state.report)


def _deepgemm_wave_reuse_conclusions_match_the_reviewed_report(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

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

    problems.run("_deepgemm_wave_reuse_conclusions_match_the_reviewed_report", check, state.report)


def _persistent_gemm_fits_reuse_conclusions_match_the_reviewed_report(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]

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

    problems.run(
        "_persistent_gemm_fits_reuse_conclusions_match_the_reviewed_report", check, state.report
    )


def _persistent_flat_states_its_precision(state, problems):
    def check(report):
        data = report
        memory = data["functions"][0]["function_records"]["memory"]
        store = next(
            call
            for call in data["functions"][0]["calls"]
            if len(call["memory"]["operands"]) > 1
            and any(operand["arg"] == "result" for operand in call["memory"]["operands"])
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

    problems.run("_persistent_flat_states_its_precision", check, state.report)


def _mesh_started_loop_keeps_an_unknown_reuse_row(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        rows = [row for row in memory["reuse_windows"] if row["buffer"] == "x"]
        assert len(rows) == 1
        row = rows[0]
        assert row["reuse_bytes"] is None
        assert row["precision"] == "unknown"

    problems.run("_mesh_started_loop_keeps_an_unknown_reuse_row", check, state.report)


def _window_datastart(state, problems):
    def check(report):
        module = DataStart
        start_bytes = 8
        precision = "upper_bound"
        memory = report["functions"][0]["function_records"]["memory"]
        window_bytes = W * D * 2
        traffic = memory["traffic"]["storage"]["gmem"]
        assert traffic["total"]["read"] == C * (start_bytes + 2 * window_bytes)
        assert traffic["per_unit"][0]["read"] == start_bytes + 2 * window_bytes
        assert _footprint_bytes(memory, "x") == (
            C * window_bytes if module is MeshStart else window_bytes
        )
        assert memory["footprint"]["precision"] == precision

    problems.run("_window_datastart", check, state.report)


def _window_meshstart(state, problems):
    def check(report):
        module = MeshStart
        start_bytes = 0
        precision = "exact"
        memory = report["functions"][0]["function_records"]["memory"]
        window_bytes = W * D * 2
        traffic = memory["traffic"]["storage"]["gmem"]
        assert traffic["total"]["read"] == C * (start_bytes + 2 * window_bytes)
        assert traffic["per_unit"][0]["read"] == start_bytes + 2 * window_bytes
        assert _footprint_bytes(memory, "x") == (
            C * window_bytes if module is MeshStart else window_bytes
        )
        assert memory["footprint"]["precision"] == precision

    problems.run("_window_meshstart", check, state.report)


def _invariant_reuse(state, problems):
    def check(result, report):
        data = report
        memory = data["functions"][0]["function_records"]["memory"]
        traffic = memory["traffic"]["storage"]["gmem"]
        assert traffic["logical"] == {"read": 64, "write": 0}
        assert traffic["total"] == {"read": 768, "write": 0}
        assert traffic["per_unit"] == [{"read": 192, "write": 0}]
        assert _footprint_bytes(memory, "x") == 16
        footprint_total = memory["footprint"]["buffers"]["x"]["gmem"]["total"]
        assert footprint_total == 192
        four_ctas_in_flight = 4 * 1024 * 16 * 1_000_000_000 // 353
        assert data["functions"][0]["function_records"]["roofline"]["memory_ns"] == -(
            -(footprint_total * 1_000_000_000) // four_ctas_in_flight
        )
        assert N // BN == 3
        assert data["functions"][0]["function_records"]["compute-cost"]["flops"]["bf16"] == {
            "logical": 32,
            "total": 384,
            "per_unit": [96],
        }
        root = build_scopes(result.module, result.function)
        scopes = tuple(dict.fromkeys((root, *root.snapshot.definition.locations.values())))
        loaded = next(
            call
            for scope in scopes
            for path, (call, _) in scope.access.accesses.get("narrow", {}).items()
            if root.at(path[:1]) is scope
            if isinstance(call.target, Reshard)
        )
        n_scope = next(
            scope
            for scope in scopes
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

    problems.run("_invariant_reuse", check, state.result, state.report)


def _resident_wave(state, problems):
    def check(report):
        resident_data = report
        resident = report["functions"][0]["function_records"]["memory"]
        assert _footprint_bytes(resident, "x") == 132 * 4 * 2
        assert resident["traffic"]["storage"]["gmem"]["total"] == {"read": 2048, "write": 0}
        assert resident["footprint"]["buffers"]["x"]["gmem"]["total"] == 2112
        assert resident_data["functions"][0]["function_records"]["roofline"]["memory_ns"] == -(
            -(2112 * 1_000_000_000) // 4_800_000_000_000
        )

    problems.run("_resident_wave", check, state.report)


def _persistent_tiled_seeded(state, problems):
    def check(report):
        data = report
        memory = data["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "a") == 12 * 64 * 32 * 2
        assert _footprint_bytes(memory, "b") == 11 * 32 * 64 * 2
        assert memory["footprint"]["precision"] == "exact"
        (product,) = (
            call
            for call in data["functions"][0]["calls"]
            if [operand["type"] for operand in call["memory"]["operands"]]
            == [
                f"bf16[{persistent_gemm_tiled.BM},{persistent_gemm_tiled.BK}] smem",
                f"bf16[{persistent_gemm_tiled.BK},{persistent_gemm_tiled.BN}] smem",
                f"f32[{persistent_gemm_tiled.BM},{persistent_gemm_tiled.BN}] rmem",
            ]
        )
        tile_bytes = persistent_gemm_tiled.BM * persistent_gemm_tiled.BN * 4
        assert product["memory"]["operands"][-1]["write"] == tile_bytes
        flops = product["compute-cost"]["flops"]["bf16"]["per_unit"]
        assert flops == [
            2 * persistent_gemm_tiled.BM * persistent_gemm_tiled.BN * persistent_gemm_tiled.BK
        ]
        operands = (
            persistent_gemm_tiled.BM * persistent_gemm_tiled.BK * 2,
            persistent_gemm_tiled.BK * persistent_gemm_tiled.BN * 2,
        )
        peaks = {peak["memory_level"]: peak["peak_bytes"] for peak in memory["peaks"]}
        assert peaks["smem"] == sum(operands), "the tiles fit in the most bytes ever live at once"

    problems.run("_persistent_tiled_seeded", check, state.report)


def _tile64(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 524288, "write": 0}
        assert _working_set_bytes(memory) == 2 * 64 * 64 * 2

    problems.run("_tile64", check, state.report)


def _tile128(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 262144, "write": 0}
        assert _working_set_bytes(memory) == 2 * 128 * 128 * 2

    problems.run("_tile128", check, state.report)


def _wave_w1x132(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert _working_set_bytes(memory) == (WAVE_BM + WAVE_C * WAVE_BN) * WAVE_BK * 2

    problems.run("_wave_w1x132", check, state.report)


def _wave_w11x12(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert (
            _working_set_bytes(memory)
            == (WAVE_OTHER * WAVE_BM + WAVE_G * WAVE_BN) * WAVE_BK * 2
            == 143_360
        )

    problems.run("_wave_w11x12", check, state.report)


def _wave_w12x11(state, problems):
    def check(report):
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert (
            _working_set_bytes(memory)
            == (WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN) * WAVE_BK * 2
            == 139_264
        )

    problems.run("_wave_w12x11", check, state.report)


def _underfilled(state, problems):
    def check(result):
        assert get_metadata(result.function, PerformanceSummaryMetadata).wave == (40, 132)
        row = 1024 * 16 * 1_000_000_000
        assert _first_duration(result, Reshard) == -(-row // (row // 353))

    problems.run("_underfilled", check, state.result)


def _sharedwindow(state, problems):
    def check(result):
        insert = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, InsertSlice)
        )
        gmem = get_metadata(insert, MemoryMetadata).traffic.storage.of("gmem")
        assert gmem.total.write == 4 * gmem.at(0).write == 4 * 256
        assert _first_duration(result, InsertSlice) == 3

    problems.run("_sharedwindow", check, state.result)


def _placedwindow(state, problems):
    def check(result):
        insert = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, InsertSlice)
        )
        gmem = get_metadata(insert, MemoryMetadata).traffic.storage.of("gmem")
        assert gmem.total.write == 4 * gmem.at(0).write == 4 * 256
        assert _first_duration(result, InsertSlice) == 7

    problems.run("_placedwindow", check, state.result)
