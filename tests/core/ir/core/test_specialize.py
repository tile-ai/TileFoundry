"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.dsl import *
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    residual_dims,
    variant_for,
)

GQA_DECODE = Path(__file__).parents[2] / "fixtures" / "placed" / "gqa_decode.py"


def _entry():
    return load_authored_ir(f"{GQA_DECODE}:GqaOnline").entry_function()


def test_the_model_is_dynamic_in_its_context_length_alone() -> None:
    """A derived extent is an expression.

    A derived extent is an expression, so it is not a second dimension the
    caller has to know how to compute.
    """
    assert set(residual_dims(_entry())) == {"ctx_len"}


def test_a_size_selects_the_one_implementation_that_covers_it() -> None:
    constants = load_namespace(str(GQA_DECODE))[0]
    SMALL_CONTEXT_T, MAX_CTX = constants["SMALL_CONTEXT_T"], constants["MAX_CTX"]
    entry = _entry()
    short = variant_for(entry, {"ctx_len": SMALL_CONTEXT_T})
    long = variant_for(entry, {"ctx_len": SMALL_CONTEXT_T + 1})

    assert short is not long
    assert [(p.dim_var, p.lo, p.hi) for p in short.specializations] == [
        ("ctx_len", 0, SMALL_CONTEXT_T)
    ]
    assert [(p.dim_var, p.lo, p.hi) for p in long.specializations] == [
        ("ctx_len", SMALL_CONTEXT_T + 1, MAX_CTX)
    ]


def test_a_size_no_implementation_covers_is_refused() -> None:
    with pytest.raises(SpecializationError, match="no variant covering"):
        variant_for(_entry(), {"ctx_len": load_namespace(str(GQA_DECODE))[0]["MAX_CTX"] + 1})


def test_choosing_an_implementation_needs_the_dimension_it_turns_on() -> None:
    """Selection reads the dimensions the variants name.

    Selection reads the dimensions the variants name. Skipping an unstated
    one would pick an implementation the caller never chose.
    """
    with pytest.raises(SpecializationError, match="was not given a size"):
        variant_for(_entry(), {"batch": 4})
