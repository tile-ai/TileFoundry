"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

import pytest

from tests.core.fixtures.placed.gqa_decode import (
    MAX_CTX,
    SMALL_CONTEXT_T,
    GqaOnline,
)
from tilefoundry.dsl import *
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    residual_dims,
    variant_for,
)

ENTRY = GqaOnline.entry_function()


def test_the_model_is_dynamic_in_its_context_length_alone() -> None:
    """A derived extent is an expression.

    A derived extent is an expression, so it is not a second dimension the
    caller has to know how to compute.
    """
    assert set(residual_dims(ENTRY)) == {"ctx_len"}


def test_a_size_selects_the_one_implementation_that_covers_it() -> None:
    short = variant_for(ENTRY, {"ctx_len": SMALL_CONTEXT_T})
    long = variant_for(ENTRY, {"ctx_len": SMALL_CONTEXT_T + 1})

    assert short is not long
    assert [(p.dim_var, p.lo, p.hi) for p in short.specializations] == [
        ("ctx_len", 0, SMALL_CONTEXT_T)
    ]
    assert [(p.dim_var, p.lo, p.hi) for p in long.specializations] == [
        ("ctx_len", SMALL_CONTEXT_T + 1, MAX_CTX)
    ]


def test_a_size_no_implementation_covers_is_refused() -> None:
    with pytest.raises(SpecializationError, match="no variant covering"):
        variant_for(ENTRY, {"ctx_len": MAX_CTX + 1})


def test_choosing_an_implementation_needs_the_dimension_it_turns_on() -> None:
    """Selection reads the dimensions the variants name.

    Selection reads the dimensions the variants name. Skipping an unstated
    one would pick an implementation the caller never chose.
    """
    with pytest.raises(SpecializationError, match="was not given a size"):
        variant_for(ENTRY, {"batch": 4})
