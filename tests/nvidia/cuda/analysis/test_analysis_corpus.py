"""One analysis per placed root, selector, and set of dimensions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from tests.core._corpus import Case, Problems, case, special
from tests.nvidia.cuda.analysis._contracts import (
    PersistentSchedule,
    _assert_persistent_schedule,
    _footprint_bytes,
    _reuse_conclusions,
    _working_set_bytes,
    assert_internal_contract,
    assert_reported_contract,
)
from tilefoundry.analysis import (
    AnalysisPrecision,
    AnalysisResult,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    analyze,
)
from tilefoundry.analysis.access import eliminate_parameters
from tilefoundry.analysis.allocation import aligned
from tilefoundry.analysis.api import _build_context
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.compute_cost import local_duration_ns
from tilefoundry.analysis.iteration_scope import IterationScope, build_scopes
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.report import render_json, report_data
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import PerformanceServiceFacts, ThroughputFacts

FAMILIES = ("compute-cost", "memory", "roofline", "performance")

_ALLOCATION_ALIGNMENT = 16

_MHA_BATCH_GMEM_WITH_8_BYTES_ALIGNMENT_PADDING = 2_099_240 + 2_097_152 + 8

_MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 2_098_196 + 1_048_576 + 12

_MHA_SHORTER_GMEM_WITH_20_BYTES_ALIGNMENT_PADDING = 1_049_612 + 524_288 + 20

_MHA_SINGLE_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 4_198_420 + 2_097_152 + 12

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


@dataclass(frozen=True)
class AnalysisCase(Case):
    dims: Mapping[str, int] | None = None
    memory_peaks: Mapping[str, int] = MappingProxyType({})
    over_bound: frozenset[str] = frozenset()
    api: bool = False
    internal: bool = False
    persistent: PersistentSchedule | None = None

    @property
    def id(self) -> str:
        shown = ",".join(f"{k}={v}" for k, v in (self.dims or {}).items()) or "static"
        return f"{self.path.stem}.{self.name}[{shown}]"

    def program(self) -> tuple[Module, Function]:
        module = self.load()
        return module, module.entry_function()


@dataclass(frozen=True)
class Analyzed:
    result: AnalysisResult | None
    report: dict | None
    scopes: tuple[IterationScope, ...] | None


def _case_source(case: AnalysisCase) -> str:
    """Name a corpus case through the same SOURCE selector the CLI accepts."""
    return f"{case.path}:{case.name}"


def _peaks_match(case, report):
    reported = report["functions"][0]["function_records"]["memory"]
    over_bound: set[str] = set()
    for peak in reported["peaks"]:
        memory_level = peak["memory_level"]
        level_lifetimes = tuple(
            lifetime
            for lifetime in reported["lifetimes"]
            if lifetime["memory_level"] == memory_level
        )
        largest_value = max(
            (lifetime["bytes"] for lifetime in level_lifetimes),
            default=0,
        )
        assert peak["peak_bytes"] >= largest_value
        aligned_live_upper = sum(
            aligned(lifetime["bytes"], _ALLOCATION_ALIGNMENT) for lifetime in level_lifetimes
        )
        if peak["peak_bytes"] > aligned_live_upper:
            over_bound.add(memory_level)
    assert over_bound == case.over_bound
    observed = {item["memory_level"]: item["peak_bytes"] for item in reported["peaks"]}
    assert observed == case.memory_peaks


def _cli_report(case, tmp_path, capsys):
    report_path = tmp_path / "analysis.json"
    command = [
        "analyze",
        _case_source(case),
        str(report_path),
        "--memory",
        "--compute-cost",
        "--roofline",
        "--performance",
        "--json",
    ]
    for name, extent in (case.dims or {}).items():
        command.extend(("--dim", f"{name}={extent}"))
    assert cli_main(command) == 0
    assert capsys.readouterr() == ("", "")
    return json.loads(report_path.read_text())


def _report(result):
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )
    return json.loads(render_json(data))


def _scopes(result):
    context = _build_context(result.module, result.function, topology_level=result.level)
    return tuple(
        scope
        for local in context
        for scope in dict.fromkeys(
            (local.scope, *local.scope.snapshot.definition.locations.values())
        )
    )


def _predicted_ns(module, dims=None) -> int:
    """What the four families together say one program takes."""
    result = analyze(
        *check_program(module, module.entry_function(), topology_level="cta", dims=dims),
        analysis=FAMILIES,
        level="cta",
    )
    summary = get_metadata(result.function, PerformanceSummaryMetadata)
    assert summary is not None
    return summary.timeline.end_ns - summary.timeline.start_ns


def _constants(stem: str, *names: str) -> tuple:
    namespace = load_namespace(str(PLACED / f"{stem}.py"))[0]
    return tuple(namespace[name] for name in names)


def _first_duration(result, kind) -> int:
    call = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, kind)
    )
    timeline = get_metadata(call, PerformanceMetadata).timeline
    return timeline.end_ns - timeline.start_ns


@dataclass(frozen=True)
class MoreWork(AnalysisCase):
    smaller: str = ""
    smaller_dims: Mapping[str, int] | None = None

    def guard(self, state, problems):
        smaller, dims, report = self.smaller, self.smaller_dims, state.report
        summary = report["functions"][0]["function_records"]["performance"]
        timeline = summary["timeline"]
        module = load_authored_ir(str(PLACED / smaller))
        assert _predicted_ns(module, dims=dims) <= timeline["end_ns"] - timeline["start_ns"]


@case
class CausalStartReuse(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "CausalStartReuse.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 64, "rmem": 16})

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]
        rows = [row for row in memory["reuse_windows"] if row["buffer"] == "x"]
        assert len(rows) == 1
        row = rows[0]
        assert row["reuse_bytes"] is None
        assert row["precision"] == "unknown"


@case
class DataStartProbe(AnalysisCase):
    path: Path = PLACED / "data_started_window.py"
    name: str | None = "DataStart.probe"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 3_104, "rmem": 16, "smem": 256})

    def guard(self, state, problems):
        report = state.report
        C, D, W, DataStart, MeshStart = _constants(
            "data_started_window", "C", "D", "W", "DataStart", "MeshStart"
        )
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


@case
class MeshStartProbe(AnalysisCase):
    path: Path = PLACED / "data_started_window.py"
    name: str | None = "MeshStart.probe"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 3_104, "rmem": 8, "smem": 256})

    def guard(self, state, problems):
        report = state.report
        C, D, W, DataStart, MeshStart = _constants(
            "data_started_window", "C", "D", "W", "DataStart", "MeshStart"
        )
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


@case
class GemmNT128W17x8(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NT128x128x64_w17x8.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 52_428_800,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
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


@case
class GemmNN64W1x132(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MK_NN64x128x32_w1x132.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
        WAVE_BM, WAVE_BN, WAVE_BK, WAVE_C = _constants(
            "gemm_schedules", "WAVE_BM", "WAVE_BN", "WAVE_BK", "WAVE_C"
        )
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert _working_set_bytes(memory) == (WAVE_BM + WAVE_C * WAVE_BN) * WAVE_BK * 2


@case
class GemmNN128W12x11K4096(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NN128x128x64_w12x11_k4096.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 67_108_864,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
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


@case
class GemmNN64W11x12(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NN64x128x32_w11x12.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
        WAVE_BM, WAVE_BN, WAVE_BK, WAVE_G, WAVE_OTHER = _constants(
            "gemm_schedules", "WAVE_BM", "WAVE_BN", "WAVE_BK", "WAVE_G", "WAVE_OTHER"
        )
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert (
            _working_set_bytes(memory)
            == (WAVE_OTHER * WAVE_BM + WAVE_G * WAVE_BN) * WAVE_BK * 2
            == 143_360
        )


@case
class GemmNN64W12x11(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NN64x128x32_w12x11.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        }
    )
    internal: bool = True

    def guard(self, state, problems):
        report = state.report
        WAVE_BM, WAVE_BN, WAVE_BK, WAVE_G, WAVE_OTHER = _constants(
            "gemm_schedules", "WAVE_BM", "WAVE_BN", "WAVE_BK", "WAVE_G", "WAVE_OTHER"
        )
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"]["read"] == 428_212_224
        assert (
            _working_set_bytes(memory)
            == (WAVE_G * WAVE_BM + WAVE_OTHER * WAVE_BN) * WAVE_BK * 2
            == 139_264
        )


@case
class GemmNN128(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NN128.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 131_072,
            "rmem": 98304,
            "smem": 128 * 128 * 2 + 128 * 128 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 262144, "write": 0}
        assert _working_set_bytes(memory) == 2 * 128 * 128 * 2


@case
class GemmNN64(AnalysisCase):
    path: Path = PLACED / "gemm_schedules.py"
    name: str | None = "Gemm_MNK_NN64.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 131_072,
            "rmem": 24576,
            "smem": 64 * 64 * 2 + 64 * 64 * 2,
        }
    )

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]
        assert memory["traffic"]["storage"]["gmem"]["total"] == {"read": 524288, "write": 0}
        assert _working_set_bytes(memory) == 2 * 64 * 64 * 2


@case
class UnderFilled(AnalysisCase):
    path: Path = PLACED / "mesh_pricing.py"
    name: str | None = "UnderFilled.f"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {"gmem": 2 * 40 * 8192 * 2, "rmem": 8192 * 2}
    )
    api: bool = True

    def guard(self, state, problems):
        result = state.result
        assert get_metadata(result.function, PerformanceSummaryMetadata).wave == (40, 132)
        row = 1024 * 16 * 1_000_000_000
        assert _first_duration(result, Reshard) == -(-row // (row // 353))


@case
class SharedWindow(AnalysisCase):
    path: Path = PLACED / "mesh_pricing.py"
    name: str | None = "SharedWindow.f"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 256 + 4 * 256, "rmem": 0})
    api: bool = True

    def guard(self, state, problems):
        result = state.result
        insert = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, InsertSlice)
        )
        gmem = get_metadata(insert, MemoryMetadata).traffic.storage.of("gmem")
        assert gmem.total.write == 4 * gmem.at(0).write == 4 * 256
        assert _first_duration(result, InsertSlice) == 3


@case
class PlacedWindow(AnalysisCase):
    path: Path = PLACED / "mesh_pricing.py"
    name: str | None = "PlacedWindow.f"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 256 + 4 * 256, "rmem": 8})
    api: bool = True

    def guard(self, state, problems):
        result = state.result
        insert = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, InsertSlice)
        )
        gmem = get_metadata(insert, MemoryMetadata).traffic.storage.of("gmem")
        assert gmem.total.write == 4 * gmem.at(0).write == 4 * 256
        assert _first_duration(result, InsertSlice) == 7


@case
class InvariantReuse(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "InvariantReuse.reuse"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 64,
            "rmem": 0,
            "smem": 32,
        }
    )
    api: bool = True

    def guard(self, state, problems):
        result, report = state.result, state.report
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
        N, BN = _constants("hand_checked", "N", "BN")
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


@case
class WaveTruncation(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "WaveTruncation.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 2_048, "rmem": 8})

    def guard(self, state, problems):
        report = state.report
        resident_data = report
        resident = report["functions"][0]["function_records"]["memory"]
        assert _footprint_bytes(resident, "x") == 132 * 4 * 2
        assert resident["traffic"]["storage"]["gmem"]["total"] == {"read": 2048, "write": 0}
        assert resident["footprint"]["buffers"]["x"]["gmem"]["total"] == 2112
        assert resident_data["functions"][0]["function_records"]["roofline"]["memory_ns"] == -(
            -(2112 * 1_000_000_000) // 4_800_000_000_000
        )


@case
class SiblingLoopReuse(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "SiblingLoopReuse.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 32, "rmem": 32})
    internal: bool = True

    def guard(self, state, problems):
        report = state.report
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


@case
class TruncatedWaveReuse(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "TruncatedWaveReuse.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 96, "rmem": 32})

    def guard(self, state, problems):
        report = state.report
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


@case
class OverlappingReads(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "OverlappingReads.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 32, "rmem": 32})

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 12 * 2
        assert memory["footprint"]["precision"] == "exact"


@case
class PackedDtype(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "PackedDtype.read"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 5, "rmem": 5})

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 5


@case
class SlicedView(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "SlicedView.read"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 64,
            "rmem": 0,
            "smem": 32,
        }
    )

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]

        assert _footprint_bytes(memory, "x") == 8 * 4
        assert set(memory["footprint"]["buffers"]) == {"x"}


@case
class StoreOnly(AnalysisCase):
    path: Path = PLACED / "hand_checked.py"
    name: str | None = "StoreOnly.store"
    memory_peaks: Mapping[str, int] = MappingProxyType({"gmem": 16, "rmem": 16})

    def guard(self, state, problems):
        report = state.report
        memory = report["functions"][0]["function_records"]["memory"]
        traffic = memory["traffic"]["storage"]["gmem"]

        assert _working_set_bytes(memory) == 8 * 2
        assert traffic["total"] == {"read": 0, "write": 16}


@case
class LongerCache(MoreWork):
    path: Path = PLACED / "mha_decode_paged.py"
    name: str | None = "LongerCache.mha_decode_paged"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": _MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING,
            "rmem": 24_840,
            "smem": 8_192,
        }
    )
    smaller: str = "mha_decode_paged.py:ShorterCache"
    smaller_dims: Mapping[str, int] | None = None


@case
class PersistentGemmFlat(AnalysisCase):
    path: Path = PLACED / "persistent_gemm_flat.py"
    name: str | None = "PersistentGemmFlat.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 130_940_928,
            "rmem": 32768,
            "smem": 64 * 32 * 2 + 32 * 64 * 2,
        }
    )
    internal: bool = True
    persistent: PersistentSchedule | None = PersistentSchedule(
        loop_trips=(("t", 30), ("ki", 128)),
        store_loop="t",
        store_precision=AnalysisPrecision.EXACT,
        compared_units=((0,), (1,)),
    )

    def guard(self, state, problems):
        report = state.report
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


@case
class PersistentGemmTiled(AnalysisCase):
    path: Path = PLACED / "persistent_gemm_tiled.py"
    name: str | None = "PersistentGemmTiled.gemm"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 130_940_928,
            "rmem": 32768,
            "smem": 64 * 32 * 2 + 32 * 64 * 2,
        }
    )
    internal: bool = True
    persistent: PersistentSchedule | None = PersistentSchedule(
        loop_trips=(("mi", 5), ("ni", 6), ("ki", 128)),
        store_loop="ni",
        store_precision=AnalysisPrecision.EXACT,
        compared_units=((0, 0), (1, 0)),
    )

    def guard(self, state, problems):
        report = state.report
        BM, BN, BK = _constants("persistent_gemm_tiled", "BM", "BN", "BK")
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
                f"bf16[{BM},{BK}] smem",
                f"bf16[{BK},{BN}] smem",
                f"f32[{BM},{BN}] rmem",
            ]
        )
        tile_bytes = BM * BN * 4
        assert product["memory"]["operands"][-1]["write"] == tile_bytes
        flops = product["compute-cost"]["flops"]["bf16"]["per_unit"]
        assert flops == [2 * BM * BN * BK]
        operands = (BM * BK * 2, BK * BN * 2)
        peaks = {peak["memory_level"]: peak["peak_bytes"] for peak in memory["peaks"]}
        assert peaks["smem"] == sum(operands), "the tiles fit in the most bytes ever live at once"


@case
class LayerDecode(AnalysisCase):
    path: Path = PLACED / "qwen3_1_7b_pd.py"
    name: str | None = "PrefillLayer.layer_decode"
    dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 128, "seq": 128})
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 152_184_080,
            "rmem": 1552,
            "smem": 65_792,
        }
    )
    internal: bool = True

    def guard(self, state, problems):
        problems.run("decode scopes", self.decode_scopes, state.scopes)
        problems.run("decode traffic", self.decode_traffic, state.report)

    def decode_scopes(self, scopes):
        rms_scopes = [
            scope
            for scope in scopes
            for path, (call, _) in scope.access.accesses.get("narrow", {}).items()
            if len(path) == 1 and scope.at(path) is scope and isinstance(call.target, RMSNorm)
        ]
        assert len(rms_scopes) == 4
        assert all(not scope.loop_scopes for scope in rms_scopes)

    def decode_traffic(self, report):
        reported = report["functions"][0]["function_records"]["memory"]
        gmem = reported["traffic"]["storage"]["gmem"]
        assert gmem["logical"]["read"] + gmem["logical"]["write"] == 102_172_180
        assert gmem["total"]["read"] + gmem["total"]["write"] == 13_678_348_800


@case
class LayerPrefill(AnalysisCase):
    path: Path = PLACED / "qwen3_1_7b_pd.py"
    name: str | None = "PrefillLayer.layer_prefill"
    dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 128, "seq": 128})
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 173417488,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        }
    )
    internal: bool = True

    def guard(self, state, problems):
        scopes = state.scopes
        cache_writes = [
            (scope, access)
            for scope in scopes
            for path, (call, accesses) in scope.access.outputs.get("narrow", {}).items()
            if len(path) == 1 and scope.at(path) is scope and isinstance(call.target, CacheUpdate)
            for access in accesses
        ]
        assert cache_writes and all(
            eliminate_parameters(access.relation, scope, access.target.leaf(), narrow=True)[1]
            is AnalysisPrecision.UPPER_BOUND
            for scope, access in cache_writes
        ), "CacheUpdate cur/width lack RangeMetadata; widest_allowed must keep writes UPPER_BOUND"


@case
class ModelCtx4608Seq1(MoreWork):
    path: Path = PLACED / "qwen3_1_7b_pd.py"
    name: str | None = "PrefillLayer.model"
    dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 4608, "seq": 1})
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 4_611_686_688,
            "rmem": 1552,
            "smem": 65_792,
        }
    )
    smaller: str = "qwen3_1_7b_pd.py:PrefillLayer"
    smaller_dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 512, "seq": 1})


@case
class ModelCtx512Seq512(MoreWork):
    path: Path = PLACED / "qwen3_1_7b_pd.py"
    name: str | None = "PrefillLayer.model"
    dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 512, "seq": 512})
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 5_269_475_856,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        }
    )
    smaller: str = "qwen3_1_7b_pd.py:PrefillLayer"
    smaller_dims: Mapping[str, int] | None = MappingProxyType({"ctx_len": 0, "seq": 512})


@case
class RmsnormQuantSeq2(AnalysisCase):
    path: Path = PLACED / "rmsnorm_quant_seq2.py"
    name: str | None = "RmsnormQuantSeq2Module.rmsnorm_quant_seq_2"
    memory_peaks: Mapping[str, int] = MappingProxyType(
        {
            "gmem": 9_312,
            "rmem": 24_584,
        }
    )
    internal: bool = True

    def guard(self, state, problems):
        result = state.result
        reshaped = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Reshape)
        )
        liveness = get_metadata(result.function, LivenessMetadata)
        assert liveness.interval_of(SSAValue(reshaped)).last_used_at == 19
        assert get_metadata(reshaped, MemoryMetadata).buffer_bytes is None
        assert not get_metadata(reshaped, MemoryMetadata).offsets


EXPECTED = {
    "cyclic_split_weight.CyclicSplitWeight.read[static]": dict(memory_peaks={"gmem": 8 * 4 // 4}),
    "cyclic_split_weight.MeshOrderWeight.read[static]": dict(memory_peaks={"gmem": 32 * 4 // 32}),
    "two_dimvar_reshape.TwoDimVarReshape.squeeze[C=128,T=128]": dict(
        memory_peaks={"gmem": 128 * 128 * 4}
    ),
    "child_matmul_target.ChildMatmul.run[static]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "child_matmul_target.ChildMatmulDirect.direct[static]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "child_matmul_target.ChildMatmulRoot.gemm[static]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "child_matmul_target.ChildMatmulRoot.gemm_on_chip[static]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "child_matmul_target.ChildMatmulRoot.gemm_staged[k_len=64]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "child_matmul_target.ChildMatmulStaged.staged[k_len=64]": dict(
        memory_peaks={
            "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
            "rmem": 16384,
            "smem": 64 * 16 * 2 + 16 * 32 * 2,
        }
    ),
    "derived_prefill.DerivedPrefill.prefill[prefill_n=64,topology_only=128]": dict(
        memory_peaks={
            "gmem": 288,
        }
    ),
    "flash_split_k_decode.FlashSplitKDecode.flash_split_k_decode[ctx=128]": dict(
        memory_peaks={
            "gmem": 526_464,
            "rmem": 128 * 4 + 8,
            "smem": 34084,
        },
        internal=True,
    ),
    "fused_boundary.Inner.run[static]": dict(memory_peaks={"rmem": 8 * 2 + 8 * 2}),
    "fused_boundary.Inner.scale[static]": dict(memory_peaks={"rmem": 256}),
    "fused_boundary.FusedBoundary.root[static]": dict(
        memory_peaks={
            "gmem": 512,
            "rmem": 8 * 2 + 8 * 4,
            "smem": 32,
        }
    ),
    "fused_boundary.FusedBoundary.stage[static]": dict(memory_peaks={"smem": 64}),
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k16384.gemm[static]": dict(
        memory_peaks={
            "gmem": 268_435_456,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        }
    ),
    "gqa_decode_cuda.GqaOnline._ctx_combine[static]": dict(memory_peaks={"gmem": 291_968}),
    "gqa_decode_cuda.GqaOnline._ctx_partials[ctx_len=128]": dict(memory_peaks={"gmem": 4_483_072}),
    "gqa_decode_cuda.GqaOnline.gqa_online_attend[ctx_len=128]": dict(
        memory_peaks={
            "gmem": 282688,
            "rmem": 0,
        }
    ),
    "hand_checked.CapacityExceeded.read[static]": dict(
        memory_peaks={
            "gmem": 1_572_864,
            "rmem": 1_572_864,
        }
    ),
    "hand_checked.TruncatedWaveReuse.view[static]": dict(memory_peaks={"gmem": 96, "rmem": 0}),
    "leaf_weights.Mod.entry[static]": dict(
        memory_peaks={
            "gmem": 51_539_608_064,
            "rmem": 0,
            "smem": 160,
        }
    ),
    "leaf_weights.Mod.leaf[static]": dict(memory_peaks={"gmem": 512, "smem": 64}),
    "leaf_weights.Mod.other[static]": dict(
        memory_peaks={
            "gmem": 51_539_608_064,
            "rmem": 0,
            "smem": 160,
        }
    ),
    "mesh_slice_start.Fixed.scan[static]": dict(
        memory_peaks={
            "gmem": 4_352,
            "rmem": 0,
            "smem": 1_408,
        }
    ),
    "mesh_slice_start.OutOfWindow.oob[static]": dict(memory_peaks={"gmem": 5_120, "rmem": 0}),
    "mesh_slice_start.Strided.scan[static]": dict(
        memory_peaks={
            "gmem": 4_352,
            "rmem": 8,
            "smem": 1_408,
        }
    ),
    "mha_decode_paged.Batch2Page256.mha_decode_paged[static]": dict(
        memory_peaks={
            "gmem": _MHA_BATCH_GMEM_WITH_8_BYTES_ALIGNMENT_PADDING,
            "rmem": 49_664,
            "smem": 16_384,
        }
    ),
    "mha_decode_paged.ShorterCache.mha_decode_paged[static]": dict(
        memory_peaks={
            "gmem": _MHA_SHORTER_GMEM_WITH_20_BYTES_ALIGNMENT_PADDING,
            "rmem": 12_548,
            "smem": 4_096,
        }
    ),
    "mha_decode_paged.SingleTokenPage128.mha_decode_paged[static]": dict(
        memory_peaks={
            "gmem": _MHA_SINGLE_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING,
            "rmem": 49_672,
            "smem": 16_384,
        }
    ),
    "moe_mega_kernel.MoEMegaKernel.experts[static]": dict(memory_peaks={"gmem": 92_160}),
    "moe_mega_kernel.MoEMegaKernel.routed_expert[static]": dict(
        memory_peaks={"gmem": (120 * 64 + 64) * 4}
    ),
    "moe_mega_kernel.MoEMegaKernel.shared_expert[static]": dict(
        memory_peaks={"gmem": (120 * 64 + 10 * 64) * 4}
    ),
    "performance_findings.Compare.kernel[static]": dict(memory_peaks={"gmem": 135_696}),
    "performance_findings.GmemSquare.kernel[static]": dict(memory_peaks={"gmem": 68_096}),
    "performance_findings.Levels.kernel[static]": dict(
        memory_peaks={
            "gmem": 2_113_536,
            "rmem": 16_384,
        }
    ),
    "performance_findings.LevelsNested.kernel[static]": dict(
        memory_peaks={
            "gmem": 2_113_536,
            "rmem": 16_384,
        }
    ),
    "performance_findings.LevelsOnOneMesh.kernel[static]": dict(
        memory_peaks={
            "gmem": 2_113_536,
            "rmem": 16_384,
        }
    ),
    "performance_findings.LocalTier.kernel[static]": dict(
        memory_peaks={"gmem": 68_096, "rmem": 512}
    ),
    "prefill_decode_attention.PrefillDecodeAttention.attend[ctx=128,seq=128]": dict(
        memory_peaks={
            "gmem": 1_310_720,
            "rmem": 128 * 128 * 4,
            "smem": 231424,
        }
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=0,seq=512]": dict(
        memory_peaks={
            "gmem": 5_269_475_856,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        }
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=1]": dict(
        memory_peaks={
            "gmem": 4_611_686_688,
            "rmem": 1552,
            "smem": 65_792,
        }
    ),
    "region_boundaries.RegionBoundaries.helper[static]": dict(
        memory_peaks={"gmem": 64, "rmem": 32}
    ),
    "region_boundaries.RegionBoundaries.run[static]": dict(
        memory_peaks={
            "gmem": 96,
            "rmem": 32,
            "smem": 32,
        }
    ),
    "rmsnorm.RmsnormModule.rmsnorm[static]": dict(memory_peaks={"gmem": 6_144, "rmem": 12_292}),
    "rmsnorm_seq2.RmsnormSeq2Module.rmsnorm_seq_2[static]": dict(
        memory_peaks={
            "gmem": 12_288,
            "rmem": 24_584,
        }
    ),
    "specialize_through_call.Direct.pick[n=128]": dict(memory_peaks={"gmem": 1_024, "smem": 128}),
    "specialize_through_call.Direct.run[n=128]": dict(memory_peaks={"gmem": 1_024, "smem": 128}),
    "specialize_through_call.ToCallee.pick[n=128]": dict(memory_peaks={"gmem": 1_024, "smem": 128}),
    "specialize_through_call.ToCallee.run[n=128]": dict(memory_peaks={"gmem": 1_024, "smem": 128}),
    "square_cuda.Model.main[static]": dict(memory_peaks={"gmem": 676, "rmem": 4}),
    "tiny_tp_decoder.DecoderLayer.decode[static]": dict(memory_peaks={"gmem": 48, "rmem": 32}),
    "tiny_tp_decoder.DecoderLayer.project[static]": dict(memory_peaks={"gmem": 96}),
    "tp_all_to_all.TransposeShard.transpose_shard[static]": dict(memory_peaks={"gmem": 256}),
    "weighted_cuda.Weighted.scaled[static]": dict(memory_peaks={"gmem": 1_348, "rmem": 8}),
}

UNANALYZED = {
    "gpu_placed_rows": "TIR only; analysis reads a HIR entry",
    "symbolic_offset": "private roots, analyzed by name in test_analysis_families",
}


def _case(key: str) -> AnalysisCase:
    """The case `{file}.{Root}.{selector}[{dim=extent,...}|static]` names."""
    head, shown = key.removesuffix("]").split("[")
    stem, name = head.split(".", 1)
    dims = None
    if shown != "static":
        dims = {dim: int(extent) for dim, extent in (item.split("=") for item in shown.split(","))}
    return AnalysisCase(path=PLACED / f"{stem}.py", name=name, dims=dims, **EXPECTED[key])


SPECIAL = special(AnalysisCase)
OWNED = sorted(EXPECTED.keys() & {case.id for case in SPECIAL})
if OWNED:
    raise ValueError(f"expectations name cases a class owns: {', '.join(OWNED)}")
CASES = tuple(sorted((*(_case(key) for key in EXPECTED), *SPECIAL), key=lambda case: case.id))
UNCOVERED = sorted(
    {path.stem for path in PLACED.glob("*.py") if path.name != "__init__.py"}
    - {case.path.stem for case in CASES}
    - UNANALYZED.keys()
)
if UNCOVERED:
    raise ValueError(f"placed files without an analysis case: {', '.join(UNCOVERED)}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_analysis_case(
    case: AnalysisCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    problems = Problems(case)
    result = None
    scopes = None
    if case.api or case.internal:
        program = problems.run("program", case.program)
        checked = problems.run("check", lambda pair: check_program(*pair, dims=case.dims), program)
        result = problems.run("analyze", lambda pair: analyze(*pair, analysis=FAMILIES), checked)
        if case.internal:
            problems.expect("checked module", lambda r, pair: r.module is pair[0], result, checked)
        report = problems.run("report", _report, result)
    else:
        report = problems.run("CLI analyze", _cli_report, case, tmp_path, capsys)
    problems.expect("families", lambda r: set(r["executed"]) == set(FAMILIES), report)
    problems.run("reported", assert_reported_contract, report)
    problems.run("peaks", _peaks_match, case, report)
    if case.internal:
        scopes = problems.run("scopes", _scopes, result)
        problems.run("internal", assert_internal_contract, result, scopes)
        if case.persistent is not None:
            problems.run("persistent", _assert_persistent_schedule, scopes, case.persistent)
    problems.run("guard", case.guard, Analyzed(result, report, scopes), problems)
    problems.dump()
