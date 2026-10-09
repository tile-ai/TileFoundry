"""Analysing a function authored for a range of sizes.

An analysis counts elements and holds them against a machine. It has no answer
for a dimension that is still a range, so the size is stated at the call and the
program that gets measured is the one at that size.

What the call accepts stays narrow: a function this Module owns. Choosing the
size happens after that, so nothing here widens which programs a Module will
answer for -- it only lets the ones it owns be asked about at a size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import isl
import pytest

from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.mha_decode_paged import LongerCache, ShorterCache
from tests.fixtures.placed.qwen3_1_7b_pd import PrefillLayer
from tests.models.corpus import ConcreteCase, placed_cases
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    AnalysisPrecision,
    AnalysisResult,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    analyze,
)
from tilefoundry.analysis.access import Access
from tilefoundry.analysis.allocation import aligned
from tilefoundry.analysis.compute_cost import local_duration_ns
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.iteration_scope import IterationScope, build_scopes, walk_scopes
from tilefoundry.analysis.liveness import analyze_liveness
from tilefoundry.analysis.report import render_json, report_data
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_namespace
from tilefoundry.ir.core import Call, describe_expr, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.specialize import (
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.types import Topology
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget, PerformanceServiceFacts, ThroughputFacts

CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "performance")
_ALLOCATION_ALIGNMENT = 16
CASES = placed_cases()
INVENTORY = [pytest.param(case, id=case.id) for case in CASES]
API_INVENTORY = frozenset(
    {
        "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]",
        "persistent_gemm_flat.PersistentGemmFlat.gemm[static]",
        "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]",
        "gemm_schedules.Gemm_MNK_NN64x128x32_w12x11.gemm[static]",
        "hand_checked.SiblingLoopReuse.read[static]",
        "flash_split_k_decode.FlashSplitKDecode.flash_split_k_decode[ctx=128]",
        "qwen3_1_7b_pd.PrefillLayer.layer_prefill[ctx_len=128,seq=128]",
        "rmsnorm_quant_seq2.RmsnormQuantSeq2Module.rmsnorm_quant_seq_2[static]",
    }
)
CLI_INVENTORY = [param for param in INVENTORY if param.id not in API_INVENTORY]
assert API_INVENTORY <= {case.id for case in CASES}

_GQA_MATERIAL_TRANSPOSE_GMEM = 280_632
_PREFILL_MATERIAL_RESHARD_AND_TRANSPOSE_SMEM = 197_632
_QWEN_LOOP_INVARIANT_VALUES_GMEM = 152_708_368
_MHA_BATCH_GMEM_WITH_8_BYTES_ALIGNMENT_PADDING = 5_245_008
_MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 4_195_376
_MHA_SHORTER_GMEM_WITH_20_BYTES_ALIGNMENT_PADDING = 2_098_216
_MHA_SINGLE_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING = 8_392_752

KNOWN_OVER_BOUND: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class _PersistentScheduleExpectation:
    loop_trips: tuple[tuple[str, int], ...]
    store_loop: str
    store_precision: AnalysisPrecision
    compared_units: tuple[tuple[int, ...], tuple[int, ...]]


EXPECTED_MEMORY_PEAKS = {
    "child_matmul_target.ChildMatmul.run[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulDirect.direct[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.child.run[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.direct.direct[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.gemm[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.gemm_on_chip[static]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.gemm_staged[k_len=64]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulRoot.staged.staged[k_len=64]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "child_matmul_target.ChildMatmulStaged.staged[k_len=64]": {
        "gmem": 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4,
        "rmem": 64 * 32 * 4,
        "smem": 64 * 16 * 2 + 16 * 32 * 2,
    },
    "data_started_window.DataStart.probe[static]": {"gmem": 4_384, "rmem": 8, "smem": 256},
    "data_started_window.MeshStart.probe[static]": {"gmem": 4_384, "rmem": 8, "smem": 256},
    "derived_prefill.DerivedPrefill.prefill[prefill_n=64,topology_only=128]": {
        "gmem": 288,
    },
    "flash_split_k_decode.FlashSplitKDecode.flash_split_k_decode[ctx=128]": {
        "gmem": 526_464,
        "rmem": 128 * 4,
        "smem": 2 * 128 * 64 * 2 + 1_044,
    },
    "fused_boundary.FusedBoundary.inner.run[static]": {"rmem": 8 * 2 + 8 * 2},
    "fused_boundary.FusedBoundary.inner.scale[static]": {"rmem": 256},
    "fused_boundary.FusedBoundary.root[static]": {
        "gmem": 512,
        "rmem": 8 * 2 + 8 * 4,
        "smem": 32,
    },
    "fused_boundary.FusedBoundary.stage[static]": {"smem": 64},
    "gemm_schedules.Gemm_MNK_NT128x128x64_w17x8.gemm[static]": {
        "gmem": 52_428_800,
        "rmem": 128 * 128 * 4,
        "smem": 128 * 64 * 2 + 64 * 128 * 2,
    },
    "gemm_schedules.Gemm_MK_NN64x128x32_w1x132.gemm[static]": {
        "gmem": 3_244_032,
        "rmem": 64 * 128 * 4,
        "smem": 64 * 32 * 2 + 32 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k4096.gemm[static]": {
        "gmem": 67_108_864,
        "rmem": 128 * 128 * 4,
        "smem": 128 * 64 * 2 + 64 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k16384.gemm[static]": {
        "gmem": 268_435_456,
        "rmem": 128 * 128 * 4,
        "smem": 128 * 64 * 2 + 64 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN64x128x32_w11x12.gemm[static]": {
        "gmem": 3_244_032,
        "rmem": 64 * 128 * 4,
        "smem": 64 * 32 * 2 + 32 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN64x128x32_w12x11.gemm[static]": {
        "gmem": 3_244_032,
        "rmem": 64 * 128 * 4,
        "smem": 64 * 32 * 2 + 32 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN128.gemm[static]": {
        "gmem": 131_072,
        "rmem": 128 * 128 * 4,
        "smem": 128 * 128 * 2 + 128 * 128 * 2,
    },
    "gemm_schedules.Gemm_MNK_NN64.gemm[static]": {
        "gmem": 131_072,
        "rmem": 64 * 64 * 4,
        "smem": 64 * 64 * 2 + 64 * 64 * 2,
    },
    "gqa_decode.GqaOnline._ctx_combine[static]": {"gmem": 291_968},
    "gqa_decode.GqaOnline._ctx_partials[ctx_len=128]": {"gmem": 4_483_072},
    "gqa_decode.GqaOnline.gqa_online_attend[ctx_len=128]": {
        "gmem": _GQA_MATERIAL_TRANSPOSE_GMEM,
        "rmem": 0,
    },
    "hand_checked.InvariantReuse.reuse[static]": {
        "gmem": 64,
        "rmem": 0,
        "smem": 16,
    },
    "hand_checked.CapacityExceeded.read[static]": {
        "gmem": 1_572_864,
        "rmem": 1_572_864,
    },
    "hand_checked.WaveTruncation.read[static]": {"gmem": 2_048, "rmem": 8},
    "hand_checked.SiblingLoopReuse.read[static]": {"gmem": 32, "rmem": 16},
    "hand_checked.TruncatedWaveReuse.read[static]": {"gmem": 96, "rmem": 32},
    "hand_checked.TruncatedWaveReuse.view[static]": {"gmem": 96, "rmem": 0},
    "hand_checked.OverlappingReads.read[static]": {"gmem": 32, "rmem": 16},
    "hand_checked.PackedDtype.read[static]": {"gmem": 5, "rmem": 5},
    "hand_checked.SlicedView.read[static]": {
        "gmem": 64,
        "rmem": 0,
        "smem": 32,
    },
    "hand_checked.StoreOnly.store[static]": {"gmem": 16, "rmem": 16},
    "leaf_weights.Mod.entry[static]": {
        "gmem": 51_539_608_064,
        "rmem": 0,
        "smem": 96,
    },
    "leaf_weights.Mod.leaf[static]": {"gmem": 512, "smem": 64},
    "leaf_weights.Mod.other[static]": {
        "gmem": 51_539_608_064,
        "rmem": 0,
        "smem": 96,
    },
    "mesh_slice_start.Fixed.scan[static]": {
        "gmem": 4_352,
        "rmem": 0,
        "smem": 1_408,
    },
    "mesh_slice_start.OutOfWindow.oob[static]": {"gmem": 5_120, "rmem": 0},
    "mesh_slice_start.Strided.scan[static]": {
        "gmem": 4_352,
        "rmem": 8,
        "smem": 1_408,
    },
    "mha_decode_paged.Batch2Page256.mha_decode_paged[static]": {
        "gmem": _MHA_BATCH_GMEM_WITH_8_BYTES_ALIGNMENT_PADDING,
        "rmem": 49_664,
        "smem": 16_384,
    },
    "mha_decode_paged.LongerCache.mha_decode_paged[static]": {
        "gmem": _MHA_LONGER_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING,
        "rmem": 24_840,
        "smem": 8_192,
    },
    "mha_decode_paged.ShorterCache.mha_decode_paged[static]": {
        "gmem": _MHA_SHORTER_GMEM_WITH_20_BYTES_ALIGNMENT_PADDING,
        "rmem": 12_548,
        "smem": 4_096,
    },
    "mha_decode_paged.SingleTokenPage128.mha_decode_paged[static]": {
        "gmem": _MHA_SINGLE_GMEM_WITH_12_BYTES_ALIGNMENT_PADDING,
        "rmem": 49_672,
        "smem": 16_384,
    },
    "moe_mega_kernel.MoEMegaKernel.experts[static]": {"gmem": 64_000},
    "moe_mega_kernel.MoEMegaKernel.routed_expert[static]": {"gmem": 61_696},
    "moe_mega_kernel.MoEMegaKernel.shared_expert[static]": {"gmem": 64_000},
    "nested_twin.Weighted.scaled[static]": {"gmem": 1_348, "rmem": 4},
    "performance_findings.Compare.kernel[static]": {"gmem": 136_192},
    "performance_findings.GmemSquare.kernel[static]": {"gmem": 68_096},
    "performance_findings.Levels.kernel[static]": {
        "gmem": 2_113_536,
        "rmem": 16_384,
    },
    "performance_findings.LevelsNested.kernel[static]": {
        "gmem": 2_113_536,
        "rmem": 16_384,
    },
    "performance_findings.LevelsOnOneMesh.kernel[static]": {
        "gmem": 2_113_536,
        "rmem": 16_384,
    },
    "performance_findings.LocalTier.kernel[static]": {"gmem": 68_096, "rmem": 512},
    "persistent_gemm_flat.PersistentGemmFlat.gemm[static]": {
        "gmem": 130_940_928,
        "rmem": 64 * 64 * 4,
        "smem": 64 * 32 * 2 + 32 * 64 * 2,
    },
    "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]": {
        "gmem": 130_940_928,
        "rmem": 64 * 64 * 4,
        "smem": 64 * 32 * 2 + 32 * 64 * 2,
    },
    "prefill_decode_attention.PrefillDecodeAttention.attend[ctx=128,seq=128]": {
        "gmem": 1_310_720,
        "rmem": 128 * 128 * 4,
        "smem": _PREFILL_MATERIAL_RESHARD_AND_TRANSPOSE_SMEM,
    },
    "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]": {
        "gmem": _QWEN_LOOP_INVARIANT_VALUES_GMEM,
        "rmem": 1_036,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.layer_prefill[ctx_len=128,seq=128]": {
        "gmem": 171_582_480,
        "rmem": 132_608,
        "smem": 3 * 128 * 128 * 2,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=0,seq=512]": {
        "gmem": 5_269_475_856,
        "rmem": 132_608,
        "smem": 3 * 128 * 128 * 2,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=4608,seq=1]": {
        "gmem": 4_611_690_784,
        "rmem": 1_036,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=1]": {
        "gmem": 4_611_690_784,
        "rmem": 1_036,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=512]": {
        "gmem": 5_269_475_856,
        "rmem": 132_608,
        "smem": 3 * 128 * 128 * 2,
    },
    "region_boundaries.RegionBoundaries.helper[static]": {"gmem": 64, "rmem": 32},
    "region_boundaries.RegionBoundaries.run[static]": {
        "gmem": 96,
        "rmem": 32,
        "smem": 32,
    },
    "rmsnorm.RmsnormModule.rmsnorm[static]": {"gmem": 6_144, "rmem": 12_292},
    "rmsnorm_quant_seq2.RmsnormQuantSeq2Module.rmsnorm_quant_seq_2[static]": {
        "gmem": 9_312,
        "rmem": 24_672,
    },
    "rmsnorm_seq2.RmsnormSeq2Module.rmsnorm_seq_2[static]": {
        "gmem": 12_288,
        "rmem": 24_584,
    },
    "specialize_through_call.Direct.pick[n=128]": {"gmem": 1_024, "smem": 64},
    "specialize_through_call.Direct.run[n=128]": {"gmem": 1_024, "smem": 64},
    "specialize_through_call.ToCallee.pick[n=128]": {"gmem": 1_024, "smem": 64},
    "specialize_through_call.ToCallee.run[n=128]": {"gmem": 1_024, "smem": 64},
    "square_cuda.Model.main[static]": {"gmem": 676, "rmem": 4},
    "tiny_tp_decoder.DecoderLayer.decode[static]": {"gmem": 48, "rmem": 16},
    "tiny_tp_decoder.DecoderLayer.project[static]": {"gmem": 128},
    "tiny_tp_decoder.TinyTPDecoderLM.layer.decode[static]": {"gmem": 48, "rmem": 16},
    "tiny_tp_decoder.TinyTPDecoderLM.layer.project[static]": {"gmem": 128},
    "tp_all_to_all.TransposeShard.transpose_shard[static]": {"gmem": 256},
    "weighted_twin.Weighted.scaled[static]": {"gmem": 1_348, "rmem": 4},
}
EXPECTED_PERSISTENT_SCHEDULES = {
    "persistent_gemm_flat.PersistentGemmFlat.gemm[static]": _PersistentScheduleExpectation(
        loop_trips=(("t", 30), ("ki", 128)),
        store_loop="t",
        store_precision=AnalysisPrecision.EXACT,
        compared_units=((0,), (1,)),
    ),
    "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]": _PersistentScheduleExpectation(
        loop_trips=(("mi", 5), ("ni", 6), ("ki", 128)),
        store_loop="ni",
        store_precision=AnalysisPrecision.EXACT,
        compared_units=((0, 0), (1, 0)),
    ),
}

assert set(EXPECTED_MEMORY_PEAKS) == {case.id for case in CASES}
assert set(EXPECTED_PERSISTENT_SCHEDULES) <= {case.id for case in CASES}


def _loop_scopes(scopes: tuple[IterationScope, ...]) -> dict[str, IterationScope]:
    return {
        scope.owner.induction_var.name: scope
        for scope in scopes
        if isinstance(scope.owner, LoopRegion)
    }


def _insert_slice_output(scope: IterationScope) -> Access:
    for call, accesses in scope.outputs.get("narrow", {}).values():
        if isinstance(call.target, InsertSlice):
            assert len(accesses) == 1
            return accesses[0]
    raise AssertionError("loop has no InsertSlice output")


def _at_unit(image: isl.set, coordinates: tuple[int, ...]) -> isl.set:
    assert image.dim(isl.dim_type.PARAM) == len(coordinates)
    for axis, coordinate in enumerate(coordinates):
        image = image.fix_si(isl.dim_type.PARAM, axis, coordinate)
    return image


def _assert_persistent_schedule(
    scopes: tuple[IterationScope, ...],
    expected: _PersistentScheduleExpectation,
) -> None:
    loops = _loop_scopes(scopes)
    for name, trips in expected.loop_trips:
        assert loops[name].trips() == trips

    store = _insert_slice_output(loops[expected.store_loop])
    assert store.precision is expected.store_precision
    written = store.relation.range()
    first, second = (_at_unit(written, unit) for unit in expected.compared_units)
    assert first.is_disjoint(second)


def _aimed():
    """The decode example, aimed at one machine."""
    return replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )


def _subject(family: str):
    """A concrete query that satisfies the selected family's readiness."""
    module = _aimed()
    return module, module.entry_function(), DIMS


def assert_reported_contract(report: dict) -> None:
    """Every performance conclusion traces back to what it was derived from.

    The prediction contains each occurrence it timed and is no faster than the
    ideal bound, and a solve that proved nothing says so. One a loop repeats is
    written once, and its last trip still lands inside the prediction that
    contains it. A loop is not an occurrence and carries no timeline of its own.
    """
    records = report["function_records"]
    summary = records["performance"]
    timeline = summary["timeline"]
    assert 0 <= timeline["start_ns"] <= timeline["end_ns"]
    assert records["memory"]["solver_status"] in ("optimal", "feasible")
    predicted_ns = timeline["end_ns"] - timeline["start_ns"]
    assert summary["waves"] > 0 and predicted_ns % summary["waves"] == 0
    assert records["roofline"]["ideal_ns"] <= predicted_ns

    timed = 0
    for call in report["calls"]:
        record = call.get("performance")
        if record is None:
            continue
        timed += 1
        occurrence = record["timeline"]
        start, end = occurrence["start_ns"], occurrence["end_ns"]
        assert timeline["start_ns"] <= start <= end <= timeline["end_ns"], call["value"]
        trips, stride = occurrence["trips"], occurrence["stride_ns"]
        assert trips >= 1, call["value"]
        assert (stride == 0) if trips == 1 else (stride >= end - start), call["value"]
        assert end + (trips - 1) * stride <= timeline["end_ns"], call["value"]
    assert bool(timed) is bool(predicted_ns)
    for loop in report["loops"]:
        assert "performance" not in loop, loop["value"]

    def nonnegative(value: object) -> None:
        """Every quantity these four families report is a count, so none is below zero.

        Work, moved bytes, placement peaks and a bound are all counts of something that
        happened or has to happen. A negative one is not a small answer but a
        derivation that ran backwards -- a projection dividing what it should have
        multiplied, or a difference taken the wrong way round -- and it would then be
        added into a total that still looks plausible.
        """
        if isinstance(value, dict):
            for item in value.values():
                nonnegative(item)
        elif isinstance(value, list):
            for item in value:
                nonnegative(item)
        elif isinstance(value, (int, float)):
            assert value >= 0, value

    nonnegative(report)
    for lifetime in records["memory"]["lifetimes"]:
        assert 0 <= lifetime["defined_at"] <= lifetime["last_used_at"]
        assert "<buffer " not in lifetime["binding"]


def assert_internal_contract(
    result: AnalysisResult, scopes: tuple[IterationScope, ...]
) -> None:
    """Each duration matches its priced work and divides the enclosing loop trips.

    An occurrence's duration is its own compute-cost record priced at the
    target's rates. One a loop repeats is written once, so its interval is that
    many of its own durations.
    """
    module_target = result.module.resolve_target()
    throughput = module_target.get_facts(ThroughputFacts)
    services = module_target.get_facts(PerformanceServiceFacts, result.level)
    for expr in collect_exprs(result.function.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        cost = get_metadata(expr, ComputeCostMetadata)
        assert cost is not None
        duration = local_duration_ns(
            cost,
            throughput,
            services,
            moved=get_metadata(expr, MemoryMetadata),
            level=result.level,
        )
        record = get_metadata(expr, PerformanceMetadata)
        if not duration:
            assert record is None
            continue
        assert record is not None
        span = record.timeline.end_ns - record.timeline.start_ns
        assert span % duration == 0, describe_expr(expr)
        runs = span // duration
        available = 1
        owner = next(
            (scope for scope in scopes if id(expr) in scope.accesses.get("narrow", {})),
            None,
        )
        if owner is not None:
            cursor = owner
            while cursor.parent is not None:
                available *= cursor.trips()
                cursor = cursor.parent
        assert 1 <= runs <= available and available % runs == 0, describe_expr(expr)
        trips = record.timeline.trips
        assert 1 <= trips <= available and available % trips == 0, describe_expr(expr)


@pytest.mark.parametrize(
    ("smaller", "larger"),
    (
        ((ShorterCache, None), (LongerCache, None)),
        (
            (PrefillLayer, {"ctx_len": 512, "seq": 1}),
            (PrefillLayer, {"ctx_len": 4608, "seq": 1}),
        ),
        (
            (PrefillLayer, {"ctx_len": 0, "seq": 512}),
            (PrefillLayer, {"ctx_len": 512, "seq": 512}),
        ),
    ),
    ids=("paged-kv", "qwen-decode-history", "qwen-prefill-history"),
)
def test_more_of_the_same_work_is_never_predicted_to_take_less_time(smaller, larger) -> None:
    """A longer cache and a longer history are more of the same program.

    Nothing here says how much longer the prediction should be: a model that got
    the direction wrong would be reporting that reading twice the cache costs
    less than reading half of it, which is the one comparison a reader makes
    without being told to.
    """
    assert _predicted_ns(*smaller) <= _predicted_ns(*larger)


def _case_source(case: ConcreteCase, tmp_path: Path) -> str:
    """Name a corpus case through the same SOURCE selector the CLI accepts."""
    identity = case.id.partition("[")[0].split(".")
    file, root, *selection = identity
    fixture = Path(__file__).parents[1] / "fixtures" / "placed" / f"{file}.py"
    namespace, _selector = load_namespace(str(fixture))
    module = namespace[root]
    try:
        module.resolve_target()
        source = fixture
    except ValueError:
        source = tmp_path / f"{file}_{root}.py"
        source.write_text(
            "from dataclasses import replace\n"
            f"from tests.fixtures.placed.{file} import {root} as authored\n"
            "from tilefoundry.target import CudaTarget\n"
            f"{root} = replace(authored, target=CudaTarget('nvidia.h200_sxm'))\n"
        )
    return f"{source}:{root}.{'.'.join(selection)}"


def _assert_reported(case: ConcreteCase, report: dict) -> None:
    """CLI and API cases share the same report checks.

    Every placed offset is aligned to the greater of 16 bytes and the element
    width (docs/spec/analysis.md, allocation). The widest dtype here is 64 bits,
    so no element is wider than 16 bytes.
    """
    assert set(report["executed"]) == set(FAMILIES)
    assert_reported_contract(report)
    reported = report["function_records"]["memory"]
    if case.id == "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]":
        gmem = reported["traffic"]["storage"]["gmem"]
        assert gmem["logical"]["read"] + gmem["logical"]["write"] == 102_172_180
        assert gmem["total"]["read"] + gmem["total"]["write"] == 13_678_348_800
    over_bound: set[tuple[str, str]] = set()
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
        aligned_live_upper = max(
            (
                sum(
                    aligned(lifetime["bytes"], _ALLOCATION_ALIGNMENT)
                    for lifetime in level_lifetimes
                    if lifetime["defined_at"] <= point <= lifetime["last_used_at"]
                )
                for point in range(
                    max(
                        (lifetime["last_used_at"] for lifetime in level_lifetimes),
                        default=-1,
                    )
                    + 1
                )
            ),
            default=0,
        )
        if peak["peak_bytes"] > aligned_live_upper:
            over_bound.add((case.id, memory_level))
    assert over_bound == {
        key for key in KNOWN_OVER_BOUND if key[0] == case.id
    }, {
        key: KNOWN_OVER_BOUND[key]
        for key in KNOWN_OVER_BOUND
        if key[0] == case.id
    }
    observed = {item["memory_level"]: item["peak_bytes"] for item in reported["peaks"]}
    assert observed == EXPECTED_MEMORY_PEAKS[case.id]


@pytest.mark.parametrize("case", CLI_INVENTORY)
def test_every_concrete_program_predicts_coherently(
    case: ConcreteCase,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every placed program, at every size and selector it exposes.

    This inventory is the whole of what these four analyses are held to: it is
    read off the directory rather than from a list beside it, so a program added
    there is asked the same questions without anyone choosing to ask. Each of
    them is asked for all four families and has to answer with a coherent
    prediction. Each case is analysed once.
    """
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
    _assert_reported(case, json.loads(report_path.read_text()))


@pytest.mark.parametrize("case", [param for param in INVENTORY if param.id in API_INVENTORY])
def test_every_internal_record_agrees(case: ConcreteCase) -> None:
    """API cases check the same report and the invariants that require scopes."""
    owner, function = case.program()
    result = analyze(owner, function, analysis=FAMILIES, dims=case.dims)
    assert result.module is owner
    report = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )
    _assert_reported(case, json.loads(render_json(report)))
    scopes = tuple(walk_scopes(build_scopes(result.module, result.function)))
    assert_internal_contract(result, scopes)
    if case.id == "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]":
        rms_scopes = [
            scope
            for scope in scopes
            for call, _ in scope.accesses["narrow"].values()
            if isinstance(call.target, RMSNorm)
        ]
        assert len(rms_scopes) == 4
        assert all(not scope.enclosing_loops() for scope in rms_scopes)
    if case.id == "qwen3_1_7b_pd.PrefillLayer.layer_prefill[ctx_len=128,seq=128]":
        cache_writes = [
            access
            for scope in scopes
            for call, accesses in scope.outputs.get("narrow", {}).values()
            if isinstance(call.target, CacheUpdate)
            for access in accesses
        ]
        assert cache_writes and all(
            access.precision is AnalysisPrecision.UPPER_BOUND for access in cache_writes
        ), "CacheUpdate cur/width lack RangeMetadata; widest_allowed must keep writes UPPER_BOUND"
    if case.id == "rmsnorm_quant_seq2.RmsnormQuantSeq2Module.rmsnorm_quant_seq_2[static]":
        reshaped = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Reshape)
        )
        liveness = analyze_liveness(result.function)
        assert (
            liveness.interval_of(reshaped.args[0]).last_used_at
            == liveness.interval_of(reshaped).last_used_at
        )
    expected_schedule = EXPECTED_PERSISTENT_SCHEDULES.get(case.id)
    if expected_schedule is not None:
        _assert_persistent_schedule(scopes, expected_schedule)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module, function, dims = _subject(family)

    result = analyze(module, function, analysis=family, dims=dims)

    assert result.metadata_types
    assert result.module is module


def _predicted_ns(module, dims=None) -> int:
    """What the four families together say one program takes."""
    result = analyze(module, module.entry_function(), analysis=FAMILIES, level="cta", dims=dims)
    summary = get_metadata(result.function, PerformanceSummaryMetadata)
    assert summary is not None
    return summary.timeline.end_ns - summary.timeline.start_ns


@pytest.mark.parametrize("family", FAMILIES)
def test_the_result_names_the_function_that_carries_the_records(family: str) -> None:
    """The records are written onto the program measured, which is the derived one.

    Handing back the symbolic input would send a reader looking for records on a
    function that has none.
    """
    module, authored, dims = _subject(family)

    result = analyze(module, authored, analysis=family, dims=dims)

    assert result.function is not authored
    assert result.function.name == authored.name
    assert residual_dims(result.function) == ()


def test_without_a_size_the_result_names_the_record_bearing_view() -> None:
    """A static input remains authored while its analysis view carries records."""
    module = QWEN3_1_7B.build()
    function = module.lookup("mlp")

    result = analyze(module, function, analysis="compute-cost")

    assert result.module is module
    assert result.function.name == function.name
    assert origin_of(result.function) is function
    assert get_metadata(result.function, ComputeCostMetadata) is not None
    assert get_metadata(function, ComputeCostMetadata) is None


def test_a_dimension_the_function_does_not_have_is_refused() -> None:
    module = _aimed()

    with pytest.raises(AnalysisError, match="no dimension named"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={**DIMS, "batch": 2},
        )


def test_a_dimension_left_unbound_is_refused() -> None:
    """Test a dimension left unbound is refused.

    Stating some other dimension is useful while the choices are being made
    and useless to an analysis, which would meet the unbound one as an extent
    that is not a number.
    """
    module = _aimed()

    with pytest.raises(AnalysisError, match="was not given a size"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={"batch": 4},
        )


def test_an_empty_or_malformed_size_is_refused_rather_than_ignored() -> None:
    """A caller who believes they stated a size must not be left believing it."""
    module = _aimed()
    entry = module.entry_function()

    with pytest.raises(AnalysisError, match="non-empty mapping"):
        analyze(module, entry, analysis="compute-cost", dims={})
    with pytest.raises(AnalysisError, match="takes an integer extent"):
        analyze(module, entry, analysis="compute-cost", dims={"ctx_len": 32.0})


def test_a_size_states_nothing_about_a_function_from_elsewhere() -> None:
    """Ownership is settled before a size is looked at.

    Ownership is settled before a size is looked at, so a foreign function
    is refused for being foreign rather than for its dimensions.
    """
    module = _aimed()
    foreign = QWEN3_1_7B.build().lookup("mlp")

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(module, foreign, analysis="compute-cost", dims=DIMS)


def test_the_entry_at_a_chosen_size_is_still_the_entry() -> None:
    """Choosing a size does not rename the entry.

    A function specialised from the entry is a different object and the same
    program, so anything that identifies the entry by name still finds it.
    """
    module = _aimed()
    variant = variant_for(module.entry_function(), DIMS)

    assert variant.name == module.entry_function().name
