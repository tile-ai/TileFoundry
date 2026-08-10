"""Dim arithmetic in a signature: bool operands, and one name anchored twice.

``DimVar`` arithmetic builds a canonical dim ``Call`` (``CTX_LEN + 1`` ->
``DimAdd``). Its rendering is covered where a dim expression reaches the printer
(``tests/ops/test_topk.py``), and the real-model corpus deliberately keeps every
shape expressed in ``ctx_len`` alone -- a step returns its own cache entry rather
than the grown cache -- so no model reaches either contract here.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types.dim import DimAdd, simplify_dim

CTX_LEN = DimVar("CTX_LEN", 1, 4097)


def test_a_bool_is_not_an_int_dim_operand() -> None:
    """``bool`` is a subclass of ``int`` in Python.

    ``bool`` is a subclass of ``int`` in Python, so both the operator surface
    and ``simplify_dim`` (which canonicalises raw ints to ``Constant(i64, v)``)
    must reject it explicitly on either side -- otherwise ``CTX_LEN + True``
    silently becomes ``CTX_LEN + 1``, and a stray bool reaches a dim ``Call`` as
    an operand.
    """
    with pytest.raises(TypeError):
        _ = CTX_LEN + True
    with pytest.raises(TypeError):
        _ = CTX_LEN + object()

    with pytest.raises(TypeError, match="bool operand"):
        simplify_dim(DimAdd, (True, CTX_LEN))
    with pytest.raises(TypeError, match="bool operand"):
        simplify_dim(DimAdd, (CTX_LEN, False))


@func
def _dim_add_consistency(
    x: Tensor[(CTX_LEN,), "bf16"],
    y: Tensor[(CTX_LEN + 1,), "bf16"],
):
    return x


def test_verify_anchors_one_name_bare_and_nested_in_a_dim_call() -> None:
    """Two params share the same ``CTX_LEN``, one directly and one nested inside ``DimAdd``.

    Two params share the same ``CTX_LEN``, one directly and one nested inside
    ``DimAdd``; the verifier's recurse-into-Call walk must anchor both and not
    report a consistency error.
    """
    nested = _dim_add_consistency.params[1].type.shape[0]
    assert isinstance(nested, Call) and isinstance(nested.target, DimAdd)
    verify_function(_dim_add_consistency)
