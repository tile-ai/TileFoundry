"""Parser ``a, b = call(...)`` — TupleType unpack.

Every decode step returns a tuple, so the corpus evaluates the multi-output
surface end to end; the cases here are the IR shape one unpack produces, the
compile-time form that binds numbers instead, and the three ways an unpack cannot
mean anything.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TupleType


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


@func
def quant_unpack(
    x: Tensor[(1, 1536), "bf16"],
) -> Tensor[(1, 1536), "fp8e4m3"]:
    x_fp8, x_scale = quant(x)
    return x_fp8


def test_tuple_unpack_emits_tuple_get_item_with_field_dtype() -> None:
    """An unpack target binds to a ``TupleGetItem`` over the producing Call.

    An unpack target binds to a ``TupleGetItem`` over the producing Call, and
    takes the dtype of *its* field rather than the tuple's first.
    """
    body = quant_unpack.body
    assert isinstance(body, Call) and isinstance(body.target, TupleGetItem)
    assert body.target.index == 0
    assert body.args[0].target.__class__.__name__ == "Quant"
    assert body.type.dtype == DType.fp8e4m3


@func
def _ret_pair(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return (add(a, b), mul(a, b))


@func
def _caller(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    s, p = _ret_pair(a, b)
    return add(s, p)  # noqa: F405


def test_a_literal_tuple_return_is_unpackable_by_a_caller() -> None:
    """Test a literal tuple return is unpackable by a caller.

    A literal tuple return folds to a core ``Tuple`` body with a ``TupleType``
    of the element field types, which is what lets a caller destructure a nested
    ``@func`` the same way it destructures a multi-output op.
    """
    assert isinstance(_ret_pair.body, Tuple), f"body is {type(_ret_pair.body).__name__}"
    assert len(_ret_pair.body.elements) == 2
    assert isinstance(_ret_pair.return_type, TupleType)
    assert len(_ret_pair.return_type.fields) == 2
    assert all(f.dtype == DType.f32 for f in _ret_pair.return_type.fields)

    picked = [
        arg.target.index
        for arg in _caller.body.args
        if isinstance(arg, Call) and isinstance(arg.target, TupleGetItem)
    ]
    assert picked == [0, 1]


_NV, _KD, _VD = 32, 128, 64


@func
def _unpacked_dims(x: Tensor[(1, 32, 128), "f32"]) -> Tensor[(1, 64, 64), "f32"]:
    nv, kd, vd = _NV, _KD, _VD
    return reshape(x, new_shape=(1, nv * kd // vd, vd))  # noqa: F405


def test_unpacking_compile_time_values_binds_the_values() -> None:
    """``nv, kd, vd = _NV, _KD, _VD`` names three numbers.

    ``nv, kd, vd = _NV, _KD, _VD`` names three numbers, so each can serve where a
    number is required — here a shape, which no ``TupleGetItem`` could.
    """
    assert _unpacked_dims.body.target.new_shape == (1, _NV * _KD // _VD, _VD)
    assert _unpacked_dims.body.type.shape == (1, 64, 64)


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
"""


_BAD_RHS = (
    _HEADER
    + """
@func
def bad_rhs(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
    p, q = add(a, b)
    return p
"""
)

_BAD_TARGETS = (
    _HEADER
    + """
@func
def bad_targets(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 1536), "fp8e4m3"]:
    {targets} = quant(x)
    return a
"""
)


def test_tuple_unpack_errors() -> None:
    """Non-TupleType RHS / arity mismatch / nested tuple target all raise."""
    with pytest.raises(VerifyError, match="tuple unpack requires RHS of TupleType"):
        import_dsl(_dedent(_BAD_RHS))

    with pytest.raises(VerifyError, match="tuple unpack arity mismatch"):
        import_dsl(_dedent(_BAD_TARGETS.format(targets="a, b, c")))

    with pytest.raises(VerifyError, match="targets must all be plain names"):
        import_dsl(_dedent(_BAD_TARGETS.format(targets="(a, b), c")))
