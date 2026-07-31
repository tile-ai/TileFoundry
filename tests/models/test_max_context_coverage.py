"""Analysing at the largest context a model admits, without holding it.

The point of asking about a size is to learn what a deployment would cost before
paying for it. A maximum-context question is the one where that matters most and
the one most easily answered dishonestly: an analysis that built the tensors it
was reasoning about would report the right number and be useless for the size
that motivated asking.

So both halves are asserted. The reported footprint has to scale with the context
-- proving the analysis reasoned at the stated length rather than at whatever it
could afford -- and the process must not have allocated anything near it.
"""

from __future__ import annotations

import tracemalloc

import pytest

from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.ir.core import get_metadata

_FAMILIES = ("compute-cost", "memory", "roofline", "timeline")


def _models_with_an_open_dimension():
    """Every corpus model that declares a context length to be asked about."""
    return [model for model in CORPUS for case in model.sized if case.dims]


def _identify(model) -> str:
    return model.id


@pytest.mark.parametrize("model", _models_with_an_open_dimension(), ids=_identify)
def test_every_analysis_answers_at_the_largest_context(model) -> None:
    """All four families, at the model's declared ceiling rather than a sample."""
    case = model.sized[0]
    selected, function = model.resolve(model.build_for(ACCEPTANCE()), case.selector)
    ceiling = dict(case.ceiling)

    for family in _FAMILIES:
        result = analyze(selected, function, analysis=family, dims=ceiling)
        assert result.function is not function, (
            "asking at a size builds the program at that size"
        )


@pytest.mark.parametrize("model", _models_with_an_open_dimension(), ids=_identify)
def test_the_largest_context_is_reasoned_about_and_not_allocated(model) -> None:
    """The analysis accounts for the full context while never holding it.

    Both halves are compared as growth, because only growth distinguishes the two
    failures. What the analysis reports must grow with the context, so an analysis
    quietly working at a length it could afford fails. What the process holds must
    not, so an analysis that materialised what it was measuring fails while its
    reported number still looked right.

    Growth rather than an absolute budget, because a budget has to come from
    somewhere and every candidate is a stand-in for this. A bound derived from the
    model's cache tensor says nothing for a short-context model, where the fixed
    cost of specialising the function alone exceeds it -- the proxy fails there
    while the property holds, which is the wrong way round.
    """
    case = model.sized[0]
    selected, function = model.resolve(model.build_for(ACCEPTANCE()), case.selector)
    ceiling = dict(case.ceiling)

    def reads(result) -> int:
        record = get_metadata(result.function, MemoryMetadata)
        assert record is not None, "the memory analysis wrote no record"
        return sum(item.read for _level, item in record.traffic)

    tracemalloc.start()
    try:
        short = analyze(selected, function, analysis="memory", dims=dict(case.dims))
        _, peak_short = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        full = analyze(selected, function, analysis="memory", dims=ceiling)
        _, peak_full = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    read_growth = reads(full) - reads(short)
    assert read_growth > 0, (
        f"{model.id}: analysing at ctx={ceiling} reports no more traffic than at "
        f"{dict(case.dims)}, so the stated length changed nothing"
    )

    # An analysis that held the context would grow with it about as fast as the
    # traffic it reports. Two orders below that is the difference between
    # reasoning about a tensor and allocating one.
    peak_growth = max(0, peak_full - peak_short)
    assert peak_growth * 100 < read_growth, (
        f"{model.id}: analysing at ctx={ceiling} grew the traced peak by "
        f"{peak_growth} bytes while reporting {read_growth} more bytes read"
    )
