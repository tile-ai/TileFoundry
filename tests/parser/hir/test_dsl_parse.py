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

from dataclasses import dataclass

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
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


@dataclass(frozen=True)
class _Cfg:
    head_dim: int = 128
    rms_eps: float = 1e-6


_CFG = _Cfg()
_NK, _KD = 16, 128


@func
def _compile_time_operands(x: Tensor[(1, 2048), "bf16"]) -> Tensor[(1, 16, 128), "bf16"]:
    key_dim = _NK * _KD
    scaled = mul(x, _CFG.head_dim ** -0.5)  # noqa: F405
    shifted = add(scaled, _CFG.rms_eps)  # noqa: F405
    return reshape(shifted, new_shape=(1, key_dim // _KD, _KD))  # noqa: F405


def test_compile_time_values_reach_an_op_as_constants() -> None:
    """``config.head_dim ** -0.5``, ``config.rms_eps`` and a body-local
    ``_NK * _KD`` are numbers by parse time: the first two arrive as bf16 scalar
    Constants, and the third serves as a shape."""
    scaled, eps = _compile_time_operands.body.args[0].args
    operand, scale = scaled.args

    assert isinstance(scale, Constant) and scale.value == pytest.approx(128 ** -0.5)
    assert isinstance(eps, Constant) and eps.value == pytest.approx(1e-6)
    assert scale.type.dtype == DType.bf16 and eps.type.dtype == DType.bf16
    assert operand.name == "x"
    assert _compile_time_operands.body.target.new_shape == (1, 16, 128)
