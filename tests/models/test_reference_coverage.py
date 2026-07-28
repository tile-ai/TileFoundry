"""Each model's executable semantics, run and compared.

The row the report shows under `Reference` has to come from the comparison
actually happening. A wiring check establishes that a reference exists and fits;
it says nothing about whether the model computes the right thing, and a report
built from wiring alone would read as though it did.

A reference states its own boundary, and boundaries differ in kind: one Function
of the model's Module for a component, or a tree walked by an orchestration method
for a whole decoder. Both run through the Evaluator; only how they are entered
differs, so that is the only thing this reads off the case.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.corpus import ModelCase, ReferenceCase
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report
from tilefoundry.evaluator import evaluate

#: The complete-decoder boundary is production-sized, so it needs a device.
_NEEDS_DEVICE = not torch.cuda.is_available()

ATOL = RTOL = 2e-3


def _selected() -> list[tuple[ModelCase, ReferenceCase]]:
    return [(model, model.reference) for model in CORPUS if model.reference]


def _cases() -> list[object]:
    return [
        pytest.param(model, reference, id=reference.id.replace("/", "-"))
        for model, reference in _selected()
    ]


def _run(model: ModelCase, reference: ReferenceCase, drawn):
    """The reference's own boundary, entered the way the case states."""
    if reference.runner is not None:
        return reference.runner(drawn)
    module = model.build()
    return evaluate(module.lookup(reference.entry), *drawn.args)


def _compared(got) -> torch.Tensor:
    """The tensor a boundary's result is judged on.

    A boundary that also hands back state -- a decode step returns the cache
    entries its caller appends -- puts the value first. The state is checked where
    it is produced; here what is asked is whether the boundary computed its output.
    """
    return got[0] if isinstance(got, tuple) else got


@pytest.mark.skipif(_NEEDS_DEVICE, reason="references run at production dimensions")
@pytest.mark.parametrize(("model", "reference"), _cases())
def test_each_model_matches_its_oracle(model: ModelCase, reference: ReferenceCase) -> None:
    drawn = reference.inputs()

    got = reference.gate.hold(
        lambda: _compared(_run(model, reference, drawn)),
        expect=AssertionError,
        label=reference.id,
    )
    if got is None:  # a gated case that failed as its gate says it must
        return

    want = reference.oracle(drawn)
    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_report_shows_a_reference_row_per_model() -> None:
    """What ran appears under `Reference`, named by its boundary's own case id."""
    fixture = ACCEPTANCE()
    collector = CoverageCollector()
    for model, reference in _selected():
        collector.record_gate(
            reference.gate,
            model=model.id,
            target=fixture.id,
            kind="reference",
            case=reference.id,
            function=reference.entry or reference.id.rsplit("/", 1)[-1],
        )

    report = build_report(collector, CORPUS)

    for model, reference in _selected():
        rows = report[model.id]["targets"][fixture.id]["reference"]
        assert [row["case"] for row in rows] == [reference.id]


@pytest.mark.skipif(_NEEDS_DEVICE, reason="references run at production dimensions")
def test_a_wrong_result_would_fail_this_comparison() -> None:
    """The comparison can fail, which the passing case above does not show.

    It was silently not comparing at all: the gate ran the boundary and returned
    nothing, so the value never reached an assertion and the row said PASS for a
    reference that had only avoided raising. Asserting that a wrong answer is
    rejected is the part that would have caught it, so it is asserted here rather
    than left to the passing case to imply.
    """
    model, reference = _selected()[0]
    drawn = reference.inputs()

    got = reference.gate.hold(
        lambda: _compared(_run(model, reference, drawn)),
        expect=AssertionError,
        label=reference.id,
    )
    assert got is not None, "an ungated boundary must hand back what it computed"

    want = reference.oracle(drawn)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            (got * 1.01).float(), want.float(), atol=ATOL, rtol=RTOL
        )
