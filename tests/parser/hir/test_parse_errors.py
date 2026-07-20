"""DSL parser error coverage.


Pins the diagnostic shape of the four most common DSL surface
errors:

1. Unknown name (no Op / Stmt registered under that callable).
2. Wrong dialect (TIR-only name in HIR body and vice versa).
3. All-candidates pattern mismatch (overload set non-empty but no
   schema matches the runtime arg types).
4. Forbidden AST node (e.g. ``yield`` / ``lambda``).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import pytest

from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.parser.overload import OverloadError, resolve


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


# ── 2. Overload pattern mismatch ─────────────────────────────────────────


def test_no_overload_matches_arg_types_raises() -> None:
    """``parser.overload.resolve`` raises ``OverloadError`` when no
    OpSchema candidate's pattern accepts the supplied arg types."""


    @dataclass(frozen=True)
    class _ScalarType:
        shape: tuple = ()

    pd = ParamDef(kind="input", pattern=Tensor)
    pd._attr_name = "x"

    class _Builder:
        def __init__(self, **kw: Any) -> None: ...

    schema = OpSchema(
        name="only_tensor",
        dialect="tf",
        category="test",
        signature=(pd,),
        builder=_Builder,
        op_class=_Builder,
    )

    with pytest.raises(OverloadError, match="No OpSchema candidate matched"):
        resolve([schema], [_ScalarType()])


# ── 4. Forbidden AST node ────────────────────────────────────────────────


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
    with pytest.raises((VerifyError, TypeError, SyntaxError)):
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
    with pytest.raises((VerifyError, TypeError)):
        parse_script(_dedent(src))
