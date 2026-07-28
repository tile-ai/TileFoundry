"""Whether each model can be asked about at a context length of our choosing.

Its own question, and its own row in the report. A model authored as one fixed
shape analyses and schedules perfectly well and has no context length to state.
Recording that under analysis would call a working thing broken; leaving it out
would hide that the model is not yet the shape the corpus is moving towards.

The blocked cases here are the record of that gap. Each one runs, has to fail,
and has to fail for the reason the registry states -- so when a model is
rewritten to be dynamic, its case starts passing and the build breaks until
somebody corrects the matrix. That is the point of writing it down.
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
    """The same model is a pass under one heading and a block under the other,
    which is the whole reason they are two headings."""
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
    blocked = {
        row["function"]: row["reason"]
        for row in section["sized"]["tested"]
        if row["status"] == "BLOCKED"
    }
    assert blocked == {"decoder_layer": "no dimension named ['ctx_len']"}
