"""Parser ``a, b = call(...)`` — TupleType unpack."""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.tensor.tuple import Tuple
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TupleType
from tilefoundry.parser.hir_parser import parse_script


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


# ── Positive: `a, b = quant(x)` emits TupleGetItem bindings ----------------


@func
def quant_unpack(
    x: Tensor[(1, 1536), "bf16"],
) -> Tensor[(1, 1536), "fp8e4m3"]:
    x_fp8, x_scale = quant(x)
    return x_fp8


def test_tuple_unpack_emits_tuple_get_item_with_field_dtype() -> None:
    fn = quant_unpack
    body = fn.body
    assert isinstance(body, Call) and isinstance(body.target, TupleGetItem)
    assert body.target.index == 0
    assert body.args[0].target.__class__.__name__ == "Quant"
    assert body.type.dtype == DType.fp8e4m3


# ── Negatives: non-TupleType RHS / arity mismatch / nested target ----------


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
"""


BAD_RHS_SRC = _HEADER + """
@func
def bad_rhs(
    a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"],
) -> Tensor[(1, 4), "f32"]:
    p, q = add(a, b)
    return p
"""


BAD_ARITY_SRC = _HEADER + """
@func
def bad_arity(
    x: Tensor[(1, 1536), "bf16"],
) -> Tensor[(1, 1536), "fp8e4m3"]:
    a, b, c = quant(x)
    return a
"""


BAD_NESTED_SRC = _HEADER + """
@func
def bad_nested(
    x: Tensor[(1, 1536), "bf16"],
) -> Tensor[(1, 1536), "fp8e4m3"]:
    (a, b), c = quant(x)
    return a
"""


def test_tuple_unpack_errors() -> None:
    """Non-TupleType RHS / arity mismatch / nested tuple target all raise."""
    with pytest.raises(VerifyError, match="tuple unpack requires RHS of TupleType"):
        parse_script(_dedent(BAD_RHS_SRC))

    with pytest.raises(VerifyError, match="tuple unpack arity mismatch"):
        parse_script(_dedent(BAD_ARITY_SRC))

    with pytest.raises(VerifyError, match="targets must all be plain names"):
        parse_script(_dedent(BAD_NESTED_SRC))


# ── Literal tuple return: `return (a, b)` / `return a, b` → TupleType body ──


@func
def _ret_paren(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return (add(a, b), mul(a, b))


@func
def _ret_bare(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return add(a, b), mul(a, b)


@pytest.mark.parametrize("fn", [_ret_paren, _ret_bare], ids=["paren", "bare"])
def test_literal_tuple_return_parses_to_tuple_type(fn) -> None:
    """Both spellings of a literal tuple return fold to an ``hir.tensor.Tuple``
    body with a ``TupleType`` return of the element field types."""
    assert isinstance(fn.body, Tuple), f"body is {type(fn.body).__name__}"
    assert len(fn.body.elements) == 2
    assert isinstance(fn.return_type, TupleType)
    assert len(fn.return_type.fields) == 2
    assert all(f.dtype == DType.f32 for f in fn.return_type.fields)
