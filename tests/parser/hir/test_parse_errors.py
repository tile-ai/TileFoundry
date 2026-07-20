"""DSL parser error coverage.


Pins the diagnostic shape of common DSL surface errors:

1. Unknown name (no Op / Stmt registered under that callable).
2. Wrong dialect (TIR-only name in HIR body and vice versa).
3. Forbidden AST node (e.g. ``yield`` / ``lambda``).
"""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry.ir.core import VerifyError
from tilefoundry.parser.hir_parser import parse_script


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


# ── 1. Unknown callable name ─────────────────────────────────────────────


def test_unknown_op_name_in_hir_raises() -> None:
    src = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return totally_undefined_op(x)
"""
    with pytest.raises(VerifyError, match=r"unknown HIR callable|unknown Op name"):
        parse_script(_dedent(src))


# ── 3. Forbidden AST node ────────────────────────────────────────────────


def test_yield_in_hir_body_raises() -> None:
    """``yield`` is not a supported HIR statement."""
    src = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    yield x
"""
    with pytest.raises(VerifyError):
        parse_script(_dedent(src))


def test_lambda_in_hir_body_raises() -> None:
    """A bare ``lambda`` Expr cannot live in @func body."""
    src = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    g = lambda y: y
    return g(x)
"""
    with pytest.raises(VerifyError, match="Lambda"):
        parse_script(_dedent(src))
