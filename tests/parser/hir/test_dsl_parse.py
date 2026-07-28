"""Tuple-literal inputs on the DSL call surface.

An op input is normally a single value; ``insert_slice.offsets`` is the one
param kind that declares a per-axis tuple literal instead. The pair of tests
below is that opening and its containment: the declaring param parses the
literal to a core ``Tuple`` of rank-0 scalars, and any other tensor input keeps
rejecting one.

The typical op-call shapes (``relu(x)`` / ``add(a, b)`` / ``matmul(a, b)`` and
the ``tf.<op>`` namespace form) are exercised by the real-model corpus, which
prints, re-imports and evaluates them. Error diagnostics for the wider DSL
surface live in ``test_parse_errors.py``.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Tuple, VerifyError
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import DType
from tilefoundry.parser.hir_parser import parse_script


@func
def _insert_slice_offset_tuple(
    dst: Tensor[(2, 8, 4), "f32"],
    upd: Tensor[(1, 3, 4), "f32"],
    p: Tensor[(), "i32"],
) -> Tensor[(2, 8, 4), "f32"]:
    return insert_slice(dst, upd, (1, p, 0))  # noqa: F405


def test_parse_insert_slice_offset_tuple() -> None:
    """The rank-N ``insert_slice`` per-axis offset argument parses to an
    core ``Tuple`` with ordered rank-0 integer scalar fields (a literal,
    a runtime scalar, a literal) — not a rank-1 offset tensor."""
    body = _insert_slice_offset_tuple.body
    assert isinstance(body, Call) and isinstance(body.target, InsertSlice)
    offsets = body.args[2]
    assert isinstance(offsets, Tuple), f"offsets is {type(offsets).__name__}, not Tuple"
    assert len(offsets.elements) == 3
    for field in offsets.type.fields:
        assert field.shape == () and field.dtype in (DType.i32, DType.i64)


def test_tuple_input_rejected_for_non_offsets_param() -> None:
    """Containment: the tuple-literal input path is open ONLY for a param that
    declares it (``insert_slice.offsets``). A tuple literal bound to any other
    op's plain tensor input keeps the pre-existing rejection."""
    bad = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return relu((x, x))
"""
    with pytest.raises(VerifyError, match="Tuple"):
        parse_script(bad)
