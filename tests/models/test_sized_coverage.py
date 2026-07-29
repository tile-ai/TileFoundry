"""Whether each model can be asked about at a context length of our choosing.

Its own question, and its own row in the report. A model authored as one fixed
shape analyses and schedules perfectly well and has no context length to state.
Recording that under analysis would call a working thing broken; leaving it out
would hide that the model is not the shape the corpus is moving towards.

A gate here is a claim about today in both directions. A blocked case has to
fail, and for the stated reason, so a model rewritten to be dynamic breaks the
build until the matrix is corrected. An ungated case has to succeed, so a model
that loses the capability breaks it the same way.
"""

from __future__ import annotations

import pytest

from tests.models.corpus import ModelCase, SizedCase, TargetFixture
from tests.models.coverage_artifact import declare
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report
from tilefoundry.analysis import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.ir.core.module import function_selectors
from tilefoundry.ir.hir.specialize import dim_vars_reached


def _selected() -> list[tuple[ModelCase, TargetFixture, SizedCase]]:
    fixture = ACCEPTANCE()
    return [(model, fixture, case) for model in CORPUS for case in model.sized]


def _cases() -> list[object]:
    return [
        pytest.param(
            model,
            fixture,
            case,
            id=case.selector,
            marks=case.gate.expected_failure(expect=AnalysisError),
        )
        for model, fixture, case in _selected()
    ]


@pytest.mark.parametrize(("model", "fixture", "case"), _cases())
def test_each_model_is_asked_at_a_size_or_says_what_stops_it(
    model: ModelCase, fixture: TargetFixture, case: SizedCase, record_property
) -> None:
    declare(
        record_property,
        model=model.model,
        target=fixture.id,
        kind="sized",
        case=case.id,
        function=case.selector,
    )
    selected, function = model.resolve(model.build_for(fixture), case.selector)

    case.gate.hold(
        lambda: analyze(
            selected, function, analysis="compute-cost", dims=dict(case.dims)
        ),
        expect=AnalysisError,
        label=case.id,
    )


def test_every_model_with_an_open_dimension_is_asked_this_question() -> None:
    """A model that answered it and a model that cannot both have to appear.
    Silence would read as the question never having been put.

    Asked only of the models the question applies to, and which those are is
    measured rather than declared: a Module is asked at a size when some function
    of it was authored over a range, so the models required to state a `sized` case
    are exactly the ones whose functions reach a `DimVar`. A recurrent mixer whose
    state is fixed-size reaches none -- there is no context length to ask it about,
    which is what a fixed-size state means and not a capability it lacks.

    Measured over the whole tree, not the root's own functions: a dynamic kernel
    moved into a child Module would otherwise stop being asked about, and the
    question would disappear along with it rather than being answered.
    """
    for model in CORPUS:
        module = model.build()
        open_dims = {
            name
            for _selector, function in function_selectors(module)
            for name in dim_vars_reached(function)
        }
        if not open_dims:
            assert not model.sized, (
                f"{model.id} states a sized case but leaves no dimension open, so "
                f"the extent it names is not a dimension the model has"
            )
            continue
        assert model.sized, (
            f"{model.id} leaves {sorted(open_dims)} open and states nothing about "
            f"being asked at a size"
        )


def test_the_report_keeps_this_apart_from_analysis() -> None:
    """Being asked at a size is reported under its own heading, whatever the
    answer is.

    The two headings exist so the answers can differ, and they still have to be
    two headings when they agree: a model that analyses and that can be asked at
    a length has answered two questions, and collapsing them once they match
    would leave nowhere to record the next model that answers only one.
    """
    fixture = ACCEPTANCE()
    collector = CoverageCollector()
    for model, _, case in _selected():
        collector.record_gate(
            case.gate,
            model=model.model,
            target=fixture.id,
            kind="sized",
            case=case.id,
            function=case.selector,
        )
    for model in CORPUS:
        for case in model.analyze:
            collector.record_gate(
                case.gate,
                model=model.model,
                target=fixture.id,
                kind="analyze",
                case=case.id,
                function=case.selector,
            )

    section = build_report(collector, CORPUS)["qwen3_1_7b"]["targets"][fixture.id]

    assert {row["status"] for row in section["analyze"]["tested"]} == {"PASS"}
    sized = {row["function"]: row["status"] for row in section["sized"]["tested"]}
    assert sized == {"decoder_layer": "PASS"}
    # Distinct headings, not one heading reported twice: the sized row names the
    # function it asked about and does not carry the analyses' rows with it.
    assert section["sized"]["tested"] is not section["analyze"]["tested"]
    assert len(section["sized"]["tested"]) == 1
