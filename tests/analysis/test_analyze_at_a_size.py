"""Analysing a function authored for a range of sizes.

An analysis counts elements and holds them against a machine. It has no answer
for a dimension that is still a range, so the size is stated at the call and the
program that gets measured is the one at that size.

What the call accepts stays narrow: a function this Module owns. Choosing the
size happens after that, so nothing here widens which programs a Module will
answer for -- it only lets the ones it owns be asked about at a size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.mha_decode_paged import LongerCache, ShorterCache
from tests.fixtures.placed.qwen3_1_7b_pd import PrefillLayer
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    AnalysisResult,
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TrafficMetadata,
    analyze,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.ir.core import describe_expr, get_metadata
from tilefoundry.ir.hir.specialize import (
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.types.shard import (
    Topology,
)
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget

CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "performance")




def _aimed():
    """The decode example, aimed at one machine."""
    return replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )


def _subject(family: str):
    """A concrete query that satisfies the selected family's readiness."""
    module = _aimed()
    return module, module.entry_function(), DIMS


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
def test_more_of_the_same_work_is_never_predicted_to_take_less_time(
    smaller, larger
) -> None:
    """A longer cache and a longer history are more of the same program.

    Nothing here says how much longer the prediction should be: a model that got
    the direction wrong would be reporting that reading twice the cache costs
    less than reading half of it, which is the one comparison a reader makes
    without being told to.
    """
    assert _predicted_ns(*smaller) <= _predicted_ns(*larger)


def _every_number_counts_something(result: AnalysisResult) -> None:
    """Every quantity these four families report is a count, so none is below zero.

    Work, bytes, a footprint and a bound are all counts of something that
    happened or has to happen. A negative one is not a small answer but a
    derivation that ran backwards -- a projection dividing what it should have
    multiplied, or a difference taken the wrong way round -- and it would then be
    added into a total that still looks plausible.
    """
    fn = result.function
    for expr in (fn, *collect_exprs(fn.body)):
        for record, rows in (
            (ComputeCostMetadata, ("flops", "flops_per_unit", "service", "service_per_unit")),
            (TrafficMetadata, ()),
            (MemoryMetadata, ()),
            (RooflineMetadata, ()),
            (PerformanceMetadata, ()),
        ):
            held = get_metadata(expr, record)
            if held is None:
                continue
            for field in rows:
                for name, value in getattr(held, field):
                    assert value >= 0, f"{describe_expr(expr)}: {field}[{name}] = {value}"
            if record is TrafficMetadata:
                for field in ("whole", "per_unit"):
                    for level, moved in getattr(held, field):
                        assert moved.read >= 0 and moved.write >= 0, (
                            f"{describe_expr(expr)}: {field}[{level}] = {moved}"
                        )
                for position, moved in enumerate(held.operands):
                    assert moved.read >= 0 and moved.write >= 0, (
                        f"{describe_expr(expr)}: operand {position} = {moved}"
                    )
            if record is MemoryMetadata:
                for level in held.footprint:
                    assert level.peak_bytes >= 0 and level.persistent_bytes >= 0
                for item in held.lifetimes:
                    assert item.bytes >= 0 and 0 <= item.defined_at <= item.last_used_at
                    assert "<buffer " not in item.binding, describe_expr(expr)
            if record is RooflineMetadata:
                assert held.ideal_ns >= 0 and held.compute_ns >= 0 and held.memory_ns >= 0
            if record is PerformanceMetadata:
                assert 0 <= held.timeline.start_ns <= held.timeline.end_ns
    for expr in collect_exprs(fn.body):
        record = get_metadata(expr, LoopFootprintMetadata)
        if record is None:
            continue
        rows = [(item.buffer, item.level) for item in record.footprints]
        assert rows == sorted(rows), describe_expr(expr)
        assert len(rows) == len(set(rows)), describe_expr(expr)
        for item in record.footprints:
            assert item.bytes >= 0 and item.device_bytes >= 0 and item.repeated_bytes >= 0
            assert "<buffer " not in item.buffer, describe_expr(expr)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module, function, dims = _subject(family)

    result = analyze(module, function, analysis=family, dims=dims)

    assert result.metadata_types
    assert result.module is module


def _predicted_ns(module, dims=None) -> int:
    """What the four families together say one program takes."""
    result = analyze(
        module, module.entry_function(), analysis=FAMILIES, level="cta", dims=dims
    )
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
