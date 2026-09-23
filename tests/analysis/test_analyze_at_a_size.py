"""Analysing a function authored for a range of sizes.

An analysis counts elements and holds them against a machine. It has no answer
for a dimension that is still a range, so the size is stated at the call and the
program that gets measured is the one at that size.

What the call accepts stays narrow: a function this Module owns. Choosing the
size happens after that, so nothing here widens which programs a Module will
answer for -- it only lets the ones it owns be asked about at a size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import isl
import pytest

from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.mha_decode_paged import LongerCache, ShorterCache
from tests.fixtures.placed.qwen3_1_7b_pd import PrefillLayer
from tests.models.corpus import ConcreteCase, placed_cases
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    AnalysisResult,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    analyze,
)
from tilefoundry.analysis.access import Access, AccessPrecision
from tilefoundry.analysis.compute_cost import local_duration_ns
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.iteration_scope import IterationScope, build_scopes, walk_scopes
from tilefoundry.ir.core import Call, describe_expr, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.specialize import (
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types.shard import (
    Topology,
)
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget, PerformanceServiceFacts, ThroughputFacts

CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "performance")
CASES = placed_cases()
INVENTORY = [pytest.param(case, id=case.id) for case in CASES]


@dataclass(frozen=True)
class _PersistentScheduleExpectation:
    loop_trips: tuple[tuple[str, int], ...]
    store_loop: str
    store_precision: AccessPrecision
    compared_units: tuple[tuple[int, ...], tuple[int, ...]]


EXPECTED_MEMORY_PEAKS = {
    "derived_prefill.DerivedPrefill.prefill[prefill_n=64,topology_only=128]": {
        "gmem": 288,
    },
    "flash_split_k_decode.FlashSplitKDecode.flash_split_k_decode[ctx=128]": {
        "gmem": 788_480,
        "rmem": 8,
        "smem": 83_592,
    },
    "fused_boundary.FusedBoundary.inner.run[static]": {"rmem": 128},
    "fused_boundary.FusedBoundary.inner.scale[static]": {"rmem": 128},
    "fused_boundary.FusedBoundary.root[static]": {
        "gmem": 512,
        "rmem": 128,
        "smem": 32,
    },
    "fused_boundary.FusedBoundary.stage[static]": {"smem": 64},
    "gemm_schedules.Gemm_MNK_NT128x128x64_w17x8.gemm[static]": {
        "gmem": 52_445_184,
        "rmem": 32_768,
        "smem": 49_152,
    },
    "gemm_schedules.Gemm_MK_NN64x128x32_w1x132.gemm[static]": {
        "gmem": 5_414_912,
        "rmem": 16_384,
        "smem": 16_384,
    },
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k4096.gemm[static]": {
        "gmem": 67_125_248,
        "rmem": 32_768,
        "smem": 49_152,
    },
    "gemm_schedules.Gemm_MNK_NN128x128x64_w12x11_k16384.gemm[static]": {
        "gmem": 268_451_840,
        "rmem": 32_768,
        "smem": 49_152,
    },
    "gemm_schedules.Gemm_MNK_NN64x128x32_w11x12.gemm[static]": {
        "gmem": 5_414_912,
        "rmem": 16_384,
        "smem": 16_384,
    },
    "gemm_schedules.Gemm_MNK_NN64x128x32_w12x11.gemm[static]": {
        "gmem": 5_414_912,
        "rmem": 16_384,
        "smem": 16_384,
    },
    "gemm_schedules.Gemm_MNK_NN128.gemm[static]": {
        "gmem": 163_840,
        "rmem": 32_768,
        "smem": 98_304,
    },
    "gemm_schedules.Gemm_MNK_NN64.gemm[static]": {
        "gmem": 139_264,
        "rmem": 8_192,
        "smem": 24_576,
    },
    "gqa_decode.GqaOnline._ctx_combine[static]": {"gmem": 291_968},
    "gqa_decode.GqaOnline._ctx_partials[ctx_len=128]": {"gmem": 5_662_720},
    "gqa_decode.GqaOnline.gqa_online_attend[ctx_len=128]": {
        "gmem": 283_752,
        "rmem": 0,
    },
    "hand_checked.InvariantReuse.reuse[static]": {
        "gmem": 80,
        "rmem": 0,
        "smem": 64,
    },
    "hand_checked.CapacityExceeded.read[static]": {
        "gmem": 1_572_864,
        "rmem": 1_572_864,
    },
    "hand_checked.WaveTruncation.read[static]": {"gmem": 2_056, "rmem": 8},
    "hand_checked.OverlappingReads.read[static]": {"gmem": 48, "rmem": 16},
    "hand_checked.PackedDtype.read[static]": {"gmem": 5, "rmem": 5},
    "hand_checked.SlicedView.read[static]": {
        "gmem": 128,
        "rmem": 0,
        "smem": 32,
    },
    "hand_checked.StoreOnly.store[static]": {"gmem": 16, "rmem": 16},
    "leaf_weights.Mod.entry[static]": {
        "gmem": 51_539_608_064,
        "rmem": 0,
        "smem": 160,
    },
    "leaf_weights.Mod.leaf[static]": {"gmem": 512, "smem": 64},
    "leaf_weights.Mod.other[static]": {
        "gmem": 51_539_608_064,
        "rmem": 0,
        "smem": 160,
    },
    "mesh_slice_start.Fixed.scan[static]": {
        "gmem": 5_120,
        "rmem": 0,
        "smem": 1_408,
    },
    "mesh_slice_start.OutOfWindow.oob[static]": {"gmem": 6_144, "rmem": 0},
    "mesh_slice_start.Strided.scan[static]": {
        "gmem": 5_120,
        "rmem": 8,
        "smem": 1_408,
    },
    "mha_decode_paged.Batch2Page256.mha_decode_paged[static]": {
        "gmem": 5_245_000,
        "rmem": 32_768,
        "smem": 16_384,
    },
    "mha_decode_paged.LongerCache.mha_decode_paged[static]": {
        "gmem": 4_195_364,
        "rmem": 16_384,
        "smem": 8_192,
    },
    "mha_decode_paged.ShorterCache.mha_decode_paged[static]": {
        "gmem": 2_098_196,
        "rmem": 8_192,
        "smem": 4_096,
    },
    "mha_decode_paged.SingleTokenPage128.mha_decode_paged[static]": {
        "gmem": 8_392_740,
        "rmem": 32_768,
        "smem": 16_384,
    },
    "moe_mega_kernel.MoEMegaKernel.experts[static]": {"gmem": 61_440},
    "moe_mega_kernel.MoEMegaKernel.routed_expert[static]": {"gmem": 61_440},
    "moe_mega_kernel.MoEMegaKernel.shared_expert[static]": {"gmem": 61_440},
    "nested_twin.Weighted.scaled[static]": {"gmem": 1_348, "rmem": 4},
    "performance_findings.Compare.kernel[static]": {"gmem": 136_208},
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
        "gmem": 195_837_952,
        "rmem": 16_384,
        "smem": 12_288,
    },
    "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]": {
        "gmem": 195_837_952,
        "rmem": 16_384,
        "smem": 12_288,
    },
    "prefill_decode_attention.PrefillDecodeAttention.attend[ctx=128,seq=128]": {
        "gmem": 1_310_720,
        "rmem": 0,
        "smem": 229_376,
    },
    "qwen3_1_7b_pd.PrefillLayer.layer_decode[ctx_len=128,seq=128]": {
        "gmem": 145_933_316,
        "rmem": 520,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.layer_prefill[ctx_len=128,seq=128]": {
        "gmem": 177_087_496,
        "rmem": 66_560,
        "smem": 131_072,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=0,seq=512]": {
        "gmem": 5_750_002_180,
        "rmem": 66_560,
        "smem": 131_072,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=4608,seq=1]": {
        "gmem": 4_763_301_384,
        "rmem": 520,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=1]": {
        "gmem": 4_763_301_384,
        "rmem": 520,
        "smem": 65_792,
    },
    "qwen3_1_7b_pd.PrefillLayer.model[ctx_len=512,seq=512]": {
        "gmem": 5_750_002_180,
        "rmem": 66_560,
        "smem": 131_072,
    },
    "region_boundaries.RegionBoundaries.helper[static]": {"gmem": 64, "rmem": 32},
    "region_boundaries.RegionBoundaries.run[static]": {
        "gmem": 64,
        "rmem": 32,
        "smem": 32,
    },
    "rmsnorm.RmsnormModule.rmsnorm[static]": {"gmem": 6_144, "rmem": 6_144},
    "rmsnorm_quant_seq2.RmsnormQuantSeq2Module.rmsnorm_quant_seq_2[static]": {
        "gmem": 9_312,
        "rmem": 12_288,
    },
    "rmsnorm_seq2.RmsnormSeq2Module.rmsnorm_seq_2[static]": {
        "gmem": 12_288,
        "rmem": 12_288,
    },
    "specialize_through_call.Direct.pick[n=128]": {"gmem": 1_024, "smem": 128},
    "specialize_through_call.Direct.run[n=128]": {"gmem": 1_024, "smem": 128},
    "specialize_through_call.ToCallee.pick[n=128]": {"gmem": 1_024, "smem": 128},
    "specialize_through_call.ToCallee.run[n=128]": {"gmem": 1_024, "smem": 128},
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
        store_precision=AccessPrecision.WIDENED,
        compared_units=((0,), (1,)),
    ),
    "persistent_gemm_tiled.PersistentGemmTiled.gemm[static]": _PersistentScheduleExpectation(
        loop_trips=(("mi", 5), ("ni", 6), ("ki", 128)),
        store_loop="ni",
        store_precision=AccessPrecision.EXACT,
        compared_units=((0, 0), (1, 0)),
    ),
}

assert set(EXPECTED_MEMORY_PEAKS) == {case.id for case in CASES}
assert set(EXPECTED_PERSISTENT_SCHEDULES) <= {case.id for case in CASES}


def _loop_scopes(result: AnalysisResult) -> dict[str, IterationScope]:
    return {
        scope.owner.induction_var.name: scope
        for scope in walk_scopes(build_scopes(result.module, result.function))
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
    result: AnalysisResult,
    expected: _PersistentScheduleExpectation,
) -> None:
    scopes = _loop_scopes(result)
    for name, trips in expected.loop_trips:
        assert scopes[name].trips() == trips

    store = _insert_slice_output(scopes[expected.store_loop])
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


def assert_performance_contract(result: AnalysisResult) -> None:
    """Every performance conclusion traces back to what it was derived from.

    The prediction contains each occurrence it timed and is no faster than the
    ideal bound. An occurrence's duration is its own compute-cost record priced
    at the target's rates, and a solve that proved nothing says so. One a loop
    repeats is written once, so its interval is that many of its own durations
    and its last trip still lands inside the prediction that contains it.
    A loop is not an occurrence and carries neither a timeline nor a placeholder
    memory record of its own.
    """
    fn = result.function
    summary = get_metadata(fn, PerformanceSummaryMetadata)
    assert summary is not None
    assert 0 <= summary.timeline.start_ns <= summary.timeline.end_ns
    placement = get_metadata(fn, RegionMemoryMetadata)
    assert placement is not None
    assert placement.solver_status in ("optimal", "feasible")
    predicted_ns = summary.timeline.end_ns - summary.timeline.start_ns
    assert summary.waves > 0 and predicted_ns % summary.waves == 0
    bound = get_metadata(fn, RooflineMetadata)
    assert bound is not None and bound.ideal_ns <= predicted_ns

    module_target = result.module.resolve_target()
    throughput = module_target.get_facts(ThroughputFacts)
    services = module_target.get_facts(PerformanceServiceFacts, result.level)
    scopes = tuple(walk_scopes(build_scopes(result.module, fn)))
    timed = 0
    for expr in collect_exprs(fn.body):
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
        timed += 1
        assert record is not None
        assert summary.timeline.start_ns <= record.timeline.start_ns
        assert record.timeline.end_ns <= summary.timeline.end_ns

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
                if cursor.is_variant(expr):
                    available *= max(1, cursor.trips())
                cursor = cursor.parent
        assert 1 <= runs <= available and available % runs == 0, describe_expr(expr)
        trips, stride = record.timeline.trips, record.timeline.stride_ns
        assert 1 <= trips <= available and available % trips == 0, describe_expr(expr)
        assert (stride == 0) if trips == 1 else (stride >= span), describe_expr(expr)
        assert record.timeline.end_ns + (trips - 1) * stride <= summary.timeline.end_ns, (
            describe_expr(expr)
        )
    assert bool(timed) is bool(predicted_ns)
    _every_number_counts_something(result)
    for expr in collect_exprs(fn.body):
        if not isinstance(expr, LoopRegion):
            continue
        assert get_metadata(expr, PerformanceMetadata) is None, describe_expr(expr)
        assert get_metadata(expr, PerformanceSummaryMetadata) is None, describe_expr(expr)


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


def _every_number_counts_something(result: AnalysisResult) -> None:
    """Every quantity these four families report is a count, so none is below zero.

    Work, moved bytes, placement peaks and a bound are all counts of something that
    happened or has to happen. A negative one is not a small answer but a
    derivation that ran backwards -- a projection dividing what it should have
    multiplied, or a difference taken the wrong way round -- and it would then be
    added into a total that still looks plausible.
    """
    fn = result.function
    for expr in (fn, *collect_exprs(fn.body)):
        for record, rows in (
            (ComputeCostMetadata, ()),
            (MemoryMetadata, ()),
            (RegionMemoryMetadata, ()),
            (RooflineMetadata, ()),
            (PerformanceMetadata, ()),
        ):
            held = get_metadata(expr, record)
            if held is None:
                continue
            for field in rows:
                value = getattr(held, field)
                if field == "flops_logical":
                    assert value >= 0, f"{describe_expr(expr)}: {field} = {value}"
                    continue
                for name, value in value:
                    assert value >= 0, f"{describe_expr(expr)}: {field}[{name}] = {value}"
            if record is ComputeCostMetadata:
                for field in ("flops", "other_ops"):
                    breakdown = getattr(held, field)
                    for name, spread in breakdown.kinds:
                        for value in (spread.logical, spread.total, *spread.per_unit):
                            assert value >= 0, f"{describe_expr(expr)}: {field}[{name}] = {value}"
            if record in (MemoryMetadata, RegionMemoryMetadata):
                for field in ("storage", "communication"):
                    breakdown = getattr(held.traffic, field)
                    for level, spread in breakdown.kinds:
                        for moved in (spread.logical, spread.total, *spread.per_unit):
                            assert moved.read >= 0 and moved.write >= 0, (
                                f"{describe_expr(expr)}: {field}[{level}] = {moved}"
                            )
            if record is MemoryMetadata:
                for position, moved in enumerate(held.operands):
                    assert moved.read >= 0 and moved.write >= 0, (
                        f"{describe_expr(expr)}: operand {position} = {moved}"
                    )
            if record is RegionMemoryMetadata:
                for level in held.peaks:
                    assert level.peak_bytes >= 0 and level.persistent_bytes >= 0
                for item in held.lifetimes:
                    assert item.bytes >= 0 and 0 <= item.defined_at <= item.last_used_at
                    assert "<buffer " not in item.binding, describe_expr(expr)
            if record is RooflineMetadata:
                assert held.ideal_ns >= 0 and held.compute_ns >= 0 and held.memory_ns >= 0
            if record is PerformanceMetadata:
                assert 0 <= held.timeline.start_ns <= held.timeline.end_ns


@pytest.mark.parametrize("case", INVENTORY)
def test_every_concrete_program_predicts_coherently(case: ConcreteCase) -> None:
    """Every placed program, at every size and selector it exposes.

    This inventory is the whole of what these four analyses are held to: it is
    read off the directory rather than from a list beside it, so a program added
    there is asked the same questions without anyone choosing to ask. Each of
    them is asked for all four families and has to answer with a coherent
    prediction.
    """
    owner, function = case.program()
    result = analyze(owner, function, analysis=FAMILIES, dims=case.dims)

    assert result.module is owner
    assert set(result.executed) == set(FAMILIES)
    assert_performance_contract(result)
    placement = get_metadata(result.function, RegionMemoryMetadata)
    assert placement is not None
    observed = {item.memory_level: item.peak_bytes for item in placement.peaks}
    assert observed == EXPECTED_MEMORY_PEAKS[case.id]
    expected_schedule = EXPECTED_PERSISTENT_SCHEDULES.get(case.id)
    if expected_schedule is not None:
        _assert_persistent_schedule(result, expected_schedule)


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
