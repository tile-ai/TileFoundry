"""Analysing and scheduling a function authored for a range of sizes.

An analysis counts elements and holds them against a machine; a solver lays
work across a level by counting it. Neither has an answer for a dimension that
is still a range, so the size is stated at the call and the program that gets
measured is the one at that size.

What the call accepts stays narrow: a function this Module owns. Choosing the
size happens after that, so nothing here widens which programs a Module will
answer for -- it only lets the ones it owns be asked about at a size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fixtures.gqa_online import MAX_CTX, GqaOnline
from tests.models.fixtures import h200_sxm
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.inspection.analysis_report import render_text, report
from tilefoundry.ir.hir.specialize import residual_dims, variant_for
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule
from tilefoundry.target import CudaTarget

#: Small enough to solve and to analyse on a CPU gate.
CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "timeline")
#: What is asked here is that a plan exists for the stated size and verifies against
#: the program of that size. The solver cannot prove this makespan optimal, so left
#: to run it spends the whole budget improving a plan whose verdict does not change.
SOLVER = ScheduleOptions(
    timeout_seconds=60, workers=4, random_seed=0, stop_at_first_solution=True
)


def _aimed():
    """The decode example, aimed at one machine."""
    return replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),))


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module = _aimed()

    result = analyze(module, module.entry_function(), analysis=family, dims=DIMS)

    assert result.metadata_types
    assert result.module is module


@pytest.mark.parametrize("family", FAMILIES)
def test_the_result_names_the_function_that_carries_the_records(family: str) -> None:
    """The records are written onto the program that was measured, and that is
    the derived one. Handing back the symbolic input would send a reader looking
    for records on a function that has none."""
    module = _aimed()
    authored = module.entry_function()

    result = analyze(module, authored, analysis=family, dims=DIMS)

    assert result.function is not authored
    assert result.function.name == authored.name
    assert residual_dims(result.function) == ()


def test_a_report_at_a_size_carries_every_family_it_ran() -> None:
    """Several analyses at one size report all of their conclusions, not one's.

    Each analysis asked about a size builds the program itself, so each annotates a
    rebuild of its own and no two share an object. A report that read one of them
    would still list every family under `executed` while showing only that family's
    numbers -- so the ones it dropped read as analyses with nothing to say.

    The same assertion at a fixed shape cannot catch this: with nothing to rebuild,
    every family annotates one object and reading any of them reads all of them.
    """
    module = _aimed()
    authored = module.entry_function()

    data = report([
        analyze(module, authored, analysis=family, dims=DIMS) for family in FAMILIES
    ])

    assert data["executed"] == list(FAMILIES)
    # compute-cost keeps no whole-function record; the other three each keep one.
    assert set(data["function_records"]) == {"memory", "roofline", "timeline"}
    assert data["totals"]["flops"], "the work totals summed to nothing"
    text = render_text(data)
    for expected in ("peak-footprint", "theoretical-bound", "theoretical-makespan"):
        assert expected in text, f"{expected} is missing from the rendered report"


def test_a_report_at_a_size_carries_the_per_call_records_of_every_family() -> None:
    """The same for the per-Call rows, which are keyed by position rather than by
    identity: two rebuilds share no Call object, so a report that matched them by
    identity would find none of the second family's.

    `compute-cost` and `timeline` because those are the two families that record on
    Calls at all -- memory records a footprint over a whole function and has nothing
    per Call to lose, so pairing it here would assert nothing.
    """
    module = _aimed()
    authored = module.entry_function()

    data = report([
        analyze(module, authored, analysis=family, dims=DIMS)
        for family in ("compute-cost", "timeline")
    ])

    families = {name for row in data["calls"] for name in row if name != "value"}
    assert families == {"compute-cost", "timeline"}, families


def test_without_a_size_the_call_is_what_it_was() -> None:
    """A statically shaped program is unaffected: the result is its own input."""
    module = QWEN3_1_7B.build_for(h200_sxm())
    function = module.lookup("mlp")

    result = analyze(module, function, analysis="compute-cost")

    assert result.function is function


def test_a_size_no_variant_covers_is_refused() -> None:
    module = _aimed()

    with pytest.raises(AnalysisError, match="no variant covering"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={"ctx_len": MAX_CTX},
        )


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
    """Stating some other dimension is useful while the choices are being made
    and useless to an analysis, which would meet the unbound one as an extent
    that is not a number."""
    module = _aimed()

    with pytest.raises(AnalysisError, match="was not given a size"):
        analyze(
            module, module.entry_function(), analysis="compute-cost",
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
    """Ownership is settled before a size is looked at, so a foreign function
    is refused for being foreign rather than for its dimensions."""
    module = _aimed()
    foreign = QWEN3_1_7B.build().lookup("mlp")

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(module, foreign, analysis="compute-cost", dims=DIMS)


def test_scheduling_at_a_stated_size_plans_and_verifies() -> None:
    """The plan is a plan for one size, and it is checked against the program of
    that size -- which is the one the result names."""
    module = _aimed()
    authored = module.entry_function()

    result = schedule(
        module, authored, topology="cta", options=SOLVER, dims=DIMS
    )

    assert result.module is module
    assert result.function is not authored
    assert residual_dims(result.function) == ()
    result.plan.verify(module, result.function, result.topology)
    assert result.plan.to_json() == result.plan.to_json()


def test_scheduling_refuses_a_size_no_variant_covers() -> None:
    module = _aimed()

    with pytest.raises(ScheduleError, match="no variant covering"):
        schedule(
            module,
            module.entry_function(),
            topology="cta",
            options=SOLVER,
            dims={"ctx_len": MAX_CTX},
        )


def test_the_entry_at_a_chosen_size_is_still_the_entry() -> None:
    """The device-wide solver admits only the entry, and it decides that by
    name: a function specialised from the entry is a different object and the
    same program."""
    module = _aimed()
    variant = variant_for(module.entry_function(), DIMS)

    assert variant.name == module.entry_function().name
    with pytest.raises(ScheduleError, match="requires the module entry"):
        schedule(
            module,
            module.lookup("_ctx_partials"),
            topology="cta",
            options=SOLVER,
            dims=DIMS,
        )
