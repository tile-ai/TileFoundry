"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli
from tilefoundry.dsl import *
from tilefoundry.interfaces import valuation

GQA_DECODE = Path(__file__).parents[2] / "fixtures" / "placed" / "gqa_decode.py"


def _entry():
    return cli.load_authored_ir(f"{GQA_DECODE}:GqaOnline").entry_function()


def test_ctx_len_sole_residual_dim() -> None:
    """A derived extent is an expression.

    A derived extent is an expression, so it is not a second dimension the
    caller has to know how to compute.
    """
    module = cli.load_authored_ir(f"{GQA_DECODE}:GqaOnline")
    entry = module.entry_function()
    assert set(valuation.residual_dims(entry, valuation.Context(module, entry))) == {"ctx_len"}


def test_size_selects_covering_variant() -> None:
    constants = cli.load_namespace(str(GQA_DECODE))[0]
    SMALL_CONTEXT_T, MAX_CTX = constants["SMALL_CONTEXT_T"], constants["MAX_CTX"]
    entry = _entry()
    short = valuation.variant_for(entry, {"ctx_len": SMALL_CONTEXT_T})
    long = valuation.variant_for(entry, {"ctx_len": SMALL_CONTEXT_T + 1})

    assert short is not long
    assert [(p.dim_var, p.lo, p.hi) for p in short.specializations] == [
        ("ctx_len", 0, SMALL_CONTEXT_T)
    ]
    assert [(p.dim_var, p.lo, p.hi) for p in long.specializations] == [
        ("ctx_len", SMALL_CONTEXT_T + 1, MAX_CTX)
    ]


def test_uncovered_size_refused() -> None:
    with pytest.raises(valuation.SpecializationError, match="no variant covering"):
        valuation.variant_for(_entry(), {"ctx_len": cli.load_namespace(str(GQA_DECODE))[0]["MAX_CTX"] + 1})


def test_selection_requires_variant_dim() -> None:
    """Selection reads the dimensions the variants name.

    Selection reads the dimensions the variants name. Skipping an unstated
    one would pick an implementation the caller never chose.
    """
    with pytest.raises(valuation.SpecializationError, match="was not given a size"):
        valuation.variant_for(_entry(), {"batch": 4})
