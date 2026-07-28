"""Every function of every model, analysed on the machine it is aimed at.

The corpus selects functions; this runs them. One case is one (model, target,
function, analysis family), and every family the target registers is run
against every function the model defines, so a family that stops working on a
function nobody thought to name still fails somebody's test.

A blocked case is a strict expectation, not a skip. It has to fail, it has to
fail for the reason the registry states, and it has to fail in every family --
a limit that lifts halfway is still news. That direction matters more than the
passing one: a skip quietly becomes permanent, while a case that starts working
and is still listed as blocked breaks the build and gets corrected.
"""

from __future__ import annotations

import pytest

from tests.models.corpus import FunctionCase, ModelCase, TargetFixture
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report, render_report
from tilefoundry.analysis import ANALYSES, analyze
from tilefoundry.analysis.api import AnalysisError


def _families(fixture: TargetFixture) -> tuple[str, ...]:
    """Every analysis the target registers, asked of the target itself."""
    return ANALYSES.selectors_for(type(fixture.target))


def _cases() -> list[tuple[ModelCase, TargetFixture, FunctionCase, str]]:
    fixture = ACCEPTANCE()
    return [
        (model, fixture, case, family)
        for model in CORPUS
        for case in model.analyze
        for family in _families(fixture)
    ]


def _identify(item: object) -> str:
    if isinstance(item, ModelCase):
        return item.id
    if isinstance(item, TargetFixture):
        return item.id
    if isinstance(item, FunctionCase):
        return item.function
    return str(item)


@pytest.mark.parametrize(
    ("model", "fixture", "case", "family"), _cases(), ids=_identify
)
def test_every_selected_function_analyses_or_says_what_stopped_it(
    model: ModelCase,
    fixture: TargetFixture,
    case: FunctionCase,
    family: str,
) -> None:
    module = model.build_for(fixture)
    function = model.function(module, case)

    case.gate.hold(
        lambda: analyze(module, function, analysis=family),
        expect=AnalysisError,
        label=f"{case.id}/{family}",
    )



def test_the_corpus_selects_every_function_its_models_define() -> None:
    """Analyze has no reason to leave a function out, so it must not."""
    for model in CORPUS:
        assert model.untested("analyze") == (), (
            f"{model.id} defines functions no analyze case selects: "
            f"{model.untested('analyze')}"
        )


def test_an_analysis_family_is_asked_of_the_target_not_written_down() -> None:
    fixture = ACCEPTANCE()
    families = _families(fixture)
    assert families, f"{fixture.id} registers no analysis"
    assert "compute-cost" in families


def test_the_report_states_the_matrix_the_registry_declares() -> None:
    """Built from the registry rather than from what other tests happened to
    record, so it says the same thing however the suite was distributed."""
    fixture = ACCEPTANCE()
    collector = CoverageCollector()
    for model, _, case, family in _cases():
        collector.record_gate(
            case.gate,
            model=model.id,
            target=fixture.id,
            kind="analyze",
            case=f"{case.id}/{family}",
            function=case.function,
        )

    section = build_report(collector, CORPUS)["qwen3_1_7b"]["targets"][fixture.id]
    assert section["analyze"]["untested"] == []
    assert {row["status"] for row in section["analyze"]["tested"]} == {"PASS"}

    text = render_report(build_report(collector, CORPUS))
    assert "qwen3_1_7b" in text
    assert fixture.id in text
