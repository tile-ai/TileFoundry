"""Whether each model can be asked about at a context length of our choosing.

Its own question, asked separately. A model authored as one fixed shape analyses
and schedules perfectly well and has no context length to state. Folding that into
analysis would call a working thing broken; leaving it out would hide that the
model is not the shape the corpus is moving towards.

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
    model: ModelCase, fixture: TargetFixture, case: SizedCase
) -> None:
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
