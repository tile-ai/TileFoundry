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
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report
from tilefoundry.analysis import analyze
from tilefoundry.analysis.errors import AnalysisError


def _selected() -> list[tuple[ModelCase, TargetFixture, SizedCase]]:
    fixture = ACCEPTANCE()
    return [(model, fixture, case) for model in CORPUS for case in model.sized]


def _cases() -> list[object]:
    return [
        pytest.param(
            model,
            fixture,
            case,
            id=case.function,
            marks=case.gate.expected_failure(expect=AnalysisError),
        )
        for model, fixture, case in _selected()
    ]


@pytest.mark.parametrize(("model", "fixture", "case"), _cases())
def test_each_model_is_asked_at_a_size_or_says_what_stops_it(
    model: ModelCase, fixture: TargetFixture, case: SizedCase
) -> None:
    module = model.build_for(fixture)
    function = module.lookup(case.function)

    case.gate.hold(
        lambda: analyze(
            module, function, analysis="compute-cost", dims=dict(case.dims)
        ),
        expect=AnalysisError,
        label=case.id,
    )


def test_every_model_is_asked_this_question() -> None:
    """A model that answered it and a model that cannot both have to appear.
    Silence would read as the question never having been put."""
    for model in CORPUS:
        assert model.sized, f"{model.id} states nothing about being asked at a size"


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
            model=model.id,
            target=fixture.id,
            kind="sized",
            case=case.id,
            function=case.function,
        )
    for model in CORPUS:
        for case in model.analyze:
            collector.record_gate(
                case.gate,
                model=model.id,
                target=fixture.id,
                kind="analyze",
                case=case.id,
                function=case.function,
            )

    section = build_report(collector, CORPUS)["qwen3_1_7b"]["targets"][fixture.id]

    assert {row["status"] for row in section["analyze"]["tested"]} == {"PASS"}
    sized = {row["function"]: row["status"] for row in section["sized"]["tested"]}
    assert sized == {"decoder_layer": "PASS"}
    # Distinct headings, not one heading reported twice: the sized row names the
    # function it asked about and does not carry the analyses' rows with it.
    assert section["sized"]["tested"] is not section["analyze"]["tested"]
    assert len(section["sized"]["tested"]) == 1
