"""One analysis per placed root, selector, and set of dimensions."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType

import pytest

from tests.core._corpus import Case, Problems, cases
from tests.nvidia.cuda.analysis._contracts import (
    PersistentSchedule,
    _assert_persistent_schedule,
    assert_internal_contract,
    assert_reported_contract,
)
from tests.nvidia.cuda.analysis._exact import (
    _deepgemm_wave_reuse_conclusions_match_the_reviewed_report,
    _invariant_reuse,
    _mesh_started_loop_keeps_an_unknown_reuse_row,
    _overlapping_reads_are_unioned_not_summed,
    _packed_dtype_rounds_up_to_whole_bytes,
    _persistent_flat_states_its_precision,
    _persistent_gemm_fits_reuse_conclusions_match_the_reviewed_report,
    _persistent_tiled_seeded,
    _placedwindow,
    _resident_wave,
    _sharedwindow,
    _sliced_view_counts_against_the_final_source,
    _store_only_still_occupies_the_cache,
    _tile64,
    _tile128,
    _time_window_excludes_a_buffer_in_a_sibling_loop,
    _truncated_wave_keeps_reuse_from_participating_boundaries,
    _underfilled,
    _wave_w1x132,
    _wave_w11x12,
    _wave_w12x11,
    _window_datastart,
    _window_meshstart,
)
from tests.nvidia.cuda.fixtures.placed.mha_decode_paged import ShorterCache
from tests.nvidia.cuda.fixtures.placed.qwen3_1_7b_pd import PrefillLayer
from tilefoundry.analysis import (
    AnalysisPrecision,
    AnalysisResult,
    MemoryMetadata,
    PerformanceSummaryMetadata,
    analyze,
)
from tilefoundry.analysis.access import eliminate_parameters
from tilefoundry.analysis.allocation import aligned
from tilefoundry.analysis.api import _build_context
from tilefoundry.analysis.check import check_program, program_dim_vars
from tilefoundry.analysis.iteration_scope import IterationScope
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.report import render_json, report_data
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_namespace
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import TopologyFacts
from tilefoundry_nvidia import CudaTarget

FAMILIES = ("compute-cost", "memory", "roofline", "performance")

_ALLOCATION_ALIGNMENT = 16

_MHA_BATCH_GMEM_WITH_8_BYTES_ALIGNMENT_PADDING = 2_099_240 + 2_097_152 + 8

_MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 2_098_196 + 1_048_576 + 12

_MHA_SHORTER_GMEM_WITH_20_BYTES_ALIGNMENT_PADDING = 1_049_612 + 524_288 + 20

_MHA_SINGLE_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 4_198_420 + 2_097_152 + 12

STATED_DIMS: "Mapping[tuple[str, str], tuple[Mapping[str, int], ...]]" = {
    ("qwen3_1_7b_pd", "model"): (
        {"ctx_len": 0, "seq": 512},
        {"ctx_len": 512, "seq": 512},
        {"ctx_len": 512, "seq": 1},
        {"ctx_len": 4608, "seq": 1},
    ),
}

_PROBE = 128


def _probe_dims(owner: Module, function: Function) -> "Mapping[str, int] | None":
    """One concrete extent per open dimension, from its own declared range."""
    declared = program_dim_vars(owner, function)
    if not declared:
        return None
    return {
        name: max(int(var.lo), min(_PROBE, int(var.hi))) for name, var in sorted(declared.items())
    }


def _published(path: Path):
    package = ".".join(path.relative_to(Path(__file__).parents[4]).with_suffix("").parts)
    sys.path.insert(0, str(path.parent))
    try:
        source = importlib.import_module(package)
    finally:
        sys.path.pop(0)
    named = [
        (name, value)
        for name in dir(source)
        if not name.startswith("_") and isinstance(value := getattr(source, name), Module)
    ]
    children = {
        id(frame.module)
        for _name, owner in named
        for frame in list_call_paths(owner).values()
        if frame.module is not owner
    }
    return tuple(
        (name, value)
        for name, value in named
        if id(value) not in children or value.target is not None
    )


def _aim(owner):
    try:
        owner.resolve_target()
    except Exception:
        owner = replace(owner, target=CudaTarget("nvidia.h200_sxm"))
    return owner


@dataclass(frozen=True)
class AnalysisCase(Case):
    root: str = ""
    selector: str = ""
    dims: Mapping[str, int] | None = None
    memory_peaks: Mapping[str, int] = MappingProxyType({})
    over_bound: frozenset[str] = frozenset()
    api: bool = False
    internal: bool = False
    persistent: PersistentSchedule | None = None

    @property
    def id(self) -> str:
        shown = ",".join(f"{k}={v}" for k, v in (self.dims or {}).items()) or "static"
        return f"{self.path.stem}.{self.root}.{self.selector}[{shown}]"

    @classmethod
    def expand(cls, path: Path, corpus: Path) -> tuple["AnalysisCase", ...]:
        expanded = []
        for name, published in _published(path):
            owner = _aim(published)
            facts = owner.resolve_target().get_facts(TopologyFacts)
            if "cta" not in tuple(t.name for t in facts.topologies):
                continue
            for selector, frame in list_call_paths(owner).items():
                if frame.module is not owner or not isinstance(frame.function, Function):
                    continue
                for dims in STATED_DIMS.get((path.stem, selector)) or (
                    _probe_dims(owner, frame.function),
                ):
                    expanded.append(
                        cls(path=path, corpus=corpus, root=name, selector=selector, dims=dims)
                    )
        return tuple(expanded)

    def program(self) -> tuple[Module, Function]:
        owner = _aim(dict(_published(self.path))[self.root])
        frame = list_call_paths(owner)[self.selector]
        return frame.module, frame.function


@dataclass(frozen=True)
class Analyzed:
    result: AnalysisResult | None
    report: dict | None
    scopes: tuple[IterationScope, ...] | None


def _case_source(case: AnalysisCase, tmp_path: Path) -> str:
    """Name a corpus case through the same SOURCE selector the CLI accepts."""
    identity = case.id.partition("[")[0].split(".")
    file, root, *selection = identity
    fixture = Path(__file__).parents[1] / "fixtures" / "placed" / f"{file}.py"
    package = "tests.nvidia.cuda.fixtures.placed"
    if not fixture.is_file():
        fixture = Path(__file__).parents[3] / "core" / "fixtures" / "placed" / f"{file}.py"
        package = "tests.core.fixtures.placed"
    namespace, _selector = load_namespace(str(fixture))
    module = namespace[root]
    try:
        module.resolve_target()
        source = fixture
    except ValueError:
        source = tmp_path / f"{file}_{root}.py"
        source.write_text(
            "from dataclasses import replace\n"
            f"from {package}.{file} import {root} as authored\n"
            "from tilefoundry_nvidia import CudaTarget\n"
            f"{root} = replace(authored, target=CudaTarget('nvidia.h200_sxm'))\n"
        )
    return f"{source}:{root}.{'.'.join(selection)}"


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
        _case_source(case, tmp_path),
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


def _decode_scopes(state, problems):
    def check(scopes):
        rms_scopes = [
            scope
            for scope in scopes
            for path, (call, _) in scope.access.accesses.get("narrow", {}).items()
            if len(path) == 1 and scope.at(path) is scope and isinstance(call.target, RMSNorm)
        ]
        assert len(rms_scopes) == 4
        assert all(not scope.loop_scopes for scope in rms_scopes)

    problems.run("_decode_scopes", check, state.scopes)


def _prefill_writes(state, problems):
    def check(scopes):
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

    problems.run("_prefill_writes", check, state.scopes)


def _reshape_liveness(state, problems):
    def check(result):
        reshaped = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Reshape)
        )
        liveness = get_metadata(result.function, LivenessMetadata)
        assert liveness.interval_of(SSAValue(reshaped)).last_used_at == 19
        assert get_metadata(reshaped, MemoryMetadata).buffer_bytes is None
        assert not get_metadata(reshaped, MemoryMetadata).offsets

    problems.run("_reshape_liveness", check, state.result)


def _decode_traffic(state, problems):
    def check(report):
        reported = report["functions"][0]["function_records"]["memory"]
        gmem = reported["traffic"]["storage"]["gmem"]
        assert gmem["logical"]["read"] + gmem["logical"]["write"] == 102_172_180
        assert gmem["total"]["read"] + gmem["total"]["write"] == 13_678_348_800

    problems.run("decode traffic", check, state.report)


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


def _more_work(smaller, dims, state, problems):
    def check(report):
        summary = report["functions"][0]["function_records"]["performance"]
        timeline = summary["timeline"]
        assert _predicted_ns(smaller, dims=dims) <= timeline["end_ns"] - timeline["start_ns"]

    problems.run("more work", check, state.report)


EXPECTED = {
    "hand_checked.CausalStartReuse.read[static]": dict(
        memory_peaks={"gmem": 64, "rmem": 16},
        hooks=(_mesh_started_loop_keeps_an_unknown_reuse_row,),
    ),
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
    "data_started_window.DataStart.probe[static]": dict(
        memory_peaks={"gmem": 3_104, "rmem": 16, "smem": 256}, hooks=(_window_datastart,)
    ),
    "data_started_window.MeshStart.probe[static]": dict(
        memory_peaks={"gmem": 3_104, "rmem": 8, "smem": 256}, hooks=(_window_meshstart,)
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
        api=True,
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
    "gemm_schedules.Gemm_MNK_NT128x128x64_w17x8.gemm[static]": dict(
        memory_peaks={
            "gmem": 52_428_800,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        },
        hooks=(_deepgemm_wave_reuse_conclusions_match_the_reviewed_report,),
    ),
    "gemm_schedules.Gemm_MK_NN64x128x32_w1x132.gemm[static]": dict(
        memory_peaks={
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        },
        hooks=(_wave_w1x132,),
    ),
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k4096.gemm[static]": dict(
        memory_peaks={
            "gmem": 67_108_864,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        },
        hooks=(_persistent_gemm_fits_reuse_conclusions_match_the_reviewed_report,),
    ),
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k16384.gemm[static]": dict(
        memory_peaks={
            "gmem": 268_435_456,
            "rmem": 98304,
            "smem": 128 * 64 * 2 + 64 * 128 * 2,
        }
    ),
    "gemm_schedules.Gemm_MNK_NN64x128x32_w11x12.gemm[static]": dict(
        memory_peaks={
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        },
        hooks=(_wave_w11x12,),
    ),
    "gemm_schedules.Gemm_MNK_NN64x128x32_w12x11.gemm[static]": dict(
        memory_peaks={
            "gmem": 3_244_032,
            "rmem": 49152,
            "smem": 64 * 32 * 2 + 32 * 128 * 2,
        },
        api=True,
        internal=True,
        hooks=(_wave_w12x11,),
    ),
    "gemm_schedules.Gemm_MNK_NN128.gemm[static]": dict(
        memory_peaks={
            "gmem": 131_072,
            "rmem": 98304,
            "smem": 128 * 128 * 2 + 128 * 128 * 2,
        },
        hooks=(_tile128,),
    ),
    "gemm_schedules.Gemm_MNK_NN64.gemm[static]": dict(
        memory_peaks={
            "gmem": 131_072,
            "rmem": 24576,
            "smem": 64 * 64 * 2 + 64 * 64 * 2,
        },
        hooks=(_tile64,),
    ),
    "gqa_decode.GqaOnline._ctx_combine[static]": dict(memory_peaks={"gmem": 291_968}),
    "gqa_decode.GqaOnline._ctx_partials[ctx_len=128]": dict(memory_peaks={"gmem": 4_483_072}),
    "gqa_decode.GqaOnline.gqa_online_attend[ctx_len=128]": dict(
        memory_peaks={
            "gmem": 282688,
            "rmem": 0,
        }
    ),
    "mesh_pricing.UnderFilled.f[static]": dict(
        memory_peaks={"gmem": 2 * 40 * 8192 * 2, "rmem": 8192 * 2}, api=True, hooks=(_underfilled,)
    ),
    "mesh_pricing.SharedWindow.f[static]": dict(
        memory_peaks={"gmem": 256 + 4 * 256, "rmem": 0}, api=True, hooks=(_sharedwindow,)
    ),
    "mesh_pricing.PlacedWindow.f[static]": dict(
        memory_peaks={"gmem": 256 + 4 * 256, "rmem": 8}, api=True, hooks=(_placedwindow,)
    ),
    "hand_checked.InvariantReuse.reuse[static]": dict(
        memory_peaks={
            "gmem": 64,
            "rmem": 0,
            "smem": 32,
        },
        api=True,
        hooks=(_invariant_reuse,),
    ),
    "hand_checked.CapacityExceeded.read[static]": dict(
        memory_peaks={
            "gmem": 1_572_864,
            "rmem": 1_572_864,
        }
    ),
    "hand_checked.WaveTruncation.read[static]": dict(
        memory_peaks={"gmem": 2_048, "rmem": 8}, hooks=(_resident_wave,)
    ),
    "hand_checked.SiblingLoopReuse.read[static]": dict(
        memory_peaks={"gmem": 32, "rmem": 32},
        api=True,
        internal=True,
        hooks=(_time_window_excludes_a_buffer_in_a_sibling_loop,),
    ),
    "hand_checked.TruncatedWaveReuse.read[static]": dict(
        memory_peaks={"gmem": 96, "rmem": 32},
        hooks=(_truncated_wave_keeps_reuse_from_participating_boundaries,),
    ),
    "hand_checked.TruncatedWaveReuse.view[static]": dict(memory_peaks={"gmem": 96, "rmem": 0}),
    "hand_checked.OverlappingReads.read[static]": dict(
        memory_peaks={"gmem": 32, "rmem": 32}, hooks=(_overlapping_reads_are_unioned_not_summed,)
    ),
    "hand_checked.PackedDtype.read[static]": dict(
        memory_peaks={"gmem": 5, "rmem": 5}, hooks=(_packed_dtype_rounds_up_to_whole_bytes,)
    ),
    "hand_checked.SlicedView.read[static]": dict(
        memory_peaks={
            "gmem": 64,
            "rmem": 0,
            "smem": 32,
        },
        hooks=(_sliced_view_counts_against_the_final_source,),
    ),
    "hand_checked.StoreOnly.store[static]": dict(
        memory_peaks={"gmem": 16, "rmem": 16}, hooks=(_store_only_still_occupies_the_cache,)
    ),
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
    "mha_decode_paged.LongerCache.mha_decode_paged[static]": dict(
        memory_peaks={
            "gmem": _MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING,
            "rmem": 24_840,
            "smem": 8_192,
        },
        hooks=(partial(_more_work, ShorterCache, None),),
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
    "persistent_gemm_flat.PersistentGemmFlat.gemm[static]": dict(
        memory_peaks={
            "gmem": 130_940_928,
            "rmem": 32768,
            "smem": 64 * 32 * 2 + 32 * 64 * 2,
        },
        api=True,
        internal=True,
        persistent=PersistentSchedule(
            loop_trips=(("t", 30), ("ki", 128)),
            store_loop="t",
            store_precision=AnalysisPrecision.EXACT,
            compared_units=((0,), (1,)),
        ),
        hooks=(_persistent_flat_states_its_precision,),
    ),
    "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]": dict(
        memory_peaks={
            "gmem": 130_940_928,
            "rmem": 32768,
            "smem": 64 * 32 * 2 + 32 * 64 * 2,
        },
        api=True,
        internal=True,
        persistent=PersistentSchedule(
            loop_trips=(("mi", 5), ("ni", 6), ("ki", 128)),
            store_loop="ni",
            store_precision=AnalysisPrecision.EXACT,
            compared_units=((0, 0), (1, 0)),
        ),
        hooks=(_persistent_tiled_seeded,),
    ),
    "prefill_decode_attention.PrefillDecodeAttention.attend[ctx=128,seq=128]": dict(
        memory_peaks={
            "gmem": 1_310_720,
            "rmem": 128 * 128 * 4,
            "smem": 231424,
        }
    ),
    "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]": dict(
        memory_peaks={
            "gmem": 152_184_080,
            "rmem": 1552,
            "smem": 65_792,
        },
        api=True,
        internal=True,
        hooks=(
            _decode_scopes,
            _decode_traffic,
        ),
    ),
    "qwen3_1_7b_pd.PrefillLayer.layer_prefill[ctx_len=128,seq=128]": dict(
        memory_peaks={
            "gmem": 173417488,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        },
        api=True,
        internal=True,
        hooks=(_prefill_writes,),
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=0,seq=512]": dict(
        memory_peaks={
            "gmem": 5_269_475_856,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        }
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=4608,seq=1]": dict(
        memory_peaks={
            "gmem": 4_611_686_688,
            "rmem": 1552,
            "smem": 65_792,
        },
        hooks=(partial(_more_work, PrefillLayer, {"ctx_len": 512, "seq": 1}),),
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=1]": dict(
        memory_peaks={
            "gmem": 4_611_686_688,
            "rmem": 1552,
            "smem": 65_792,
        }
    ),
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=512]": dict(
        memory_peaks={
            "gmem": 5_269_475_856,
            "rmem": 198656,
            "smem": 3 * 128 * 128 * 2,
        },
        hooks=(partial(_more_work, PrefillLayer, {"ctx_len": 0, "seq": 512}),),
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
    "rmsnorm_quant_seq2.RmsnormQuantSeq2Module.rmsnorm_quant_seq_2[static]": dict(
        memory_peaks={
            "gmem": 9_312,
            "rmem": 24_584,
        },
        api=True,
        internal=True,
        hooks=(_reshape_liveness,),
    ),
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
    "tp_all_to_all.TransposeShard.transpose_shard[static]": dict(memory_peaks={"gmem": 128}),
    "weighted_twin.Weighted.scaled[static]": dict(memory_peaks={"gmem": 1_348, "rmem": 8}),
}

CORE_PLACED = Path(__file__).parents[3] / "core" / "fixtures" / "placed"
NVIDIA_PLACED = Path(__file__).parents[1] / "fixtures" / "placed"
CASES = cases(CORE_PLACED, NVIDIA_PLACED, case_type=AnalysisCase, expected=EXPECTED)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_analysis_case(
    case: AnalysisCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    problems = Problems(case)
    result = None
    scopes = None
    if case.api:
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
    state = Analyzed(result, report, scopes)
    for hook in case.hooks:
        problems.run(getattr(hook, "__name__", "more work"), hook, state, problems)
    problems.dump()
