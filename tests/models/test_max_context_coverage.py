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
    module = model.build_for(ACCEPTANCE())
    case = model.sized[0]
    function = module.lookup(case.function)
    ceiling = {name: model.namespace["config"].max_ctx for name in case.dims}

    for family in _FAMILIES:
        result = analyze(module, function, analysis=family, dims=ceiling)
        assert result.function is not function, (
            "asking at a size builds the program at that size"
        )


@pytest.mark.parametrize("model", _models_with_an_open_dimension(), ids=_identify)
def test_the_largest_context_is_reasoned_about_and_not_allocated(model) -> None:
    """The analysis accounts for the full context while never holding it.

    The bound is the model's own cache tensor at that context. Anything that
    materialised even one of those would cross it, and the reported traffic would
    still look right -- which is why the reported number alone is not the test.
    """
    module = model.build_for(ACCEPTANCE())
    case = model.sized[0]
    function = module.lookup(case.function)
    shape = model.namespace["config"]
    ceiling = {name: shape.max_ctx for name in case.dims}

    one_cache_bytes = shape.max_ctx * shape.n_kv_heads * shape.head_dim * 4

    tracemalloc.start()
    try:
        short = analyze(module, function, analysis="memory", dims=dict(case.dims))
        tracemalloc.reset_peak()
        full = analyze(module, function, analysis="memory", dims=ceiling)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    def read_bytes(result) -> int:
        """Every byte the analysis says the program reads."""
        record = get_metadata(result.function, MemoryMetadata)
        assert record is not None, "the memory analysis wrote no record"
        return sum(item.read_bytes for _level, item in record.traffic)

    # Reasoned about: the longer context reads strictly more than the short one.
    assert read_bytes(full) > read_bytes(short)
    # Not allocated: nowhere near one cache tensor at that context.
    assert peak < one_cache_bytes // 4, (
        f"analysing at ctx={shape.max_ctx} peaked at {peak} bytes, "
        f"against a single cache tensor of {one_cache_bytes}"
    )
