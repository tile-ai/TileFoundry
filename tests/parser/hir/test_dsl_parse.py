"""Tuple-literal inputs on the DSL call surface.

Most op inputs are single values; ``insert_slice.offsets`` declares a per-axis
tuple literal. These tests pin its parsing to a core ``Tuple`` of rank-0 scalars
and rejection for other tensor inputs. The model corpus covers ordinary and
``tf.<op>`` calls through print, re-import, and evaluation. Wider diagnostics
live in ``test_parse_errors.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Constant, Tuple
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import DType


@func
def _insert_slice_offset_tuple(
    dst: Tensor[(2, 8, 4), "f32"],
    upd: Tensor[(1, 3, 4), "f32"],
    p: Tensor[(), "i32"],
) -> Tensor[(2, 8, 4), "f32"]:
    return insert_slice(dst, upd, (1, p, 0))  # noqa: F405


def test_parse_insert_slice_offset_tuple() -> None:
    """Test parse insert slice offset tuple.

    The rank-N ``insert_slice`` per-axis offset argument parses to an
    core ``Tuple`` with ordered rank-0 integer scalar fields (a literal,
    a runtime scalar, a literal) — not a rank-1 offset tensor.
    """
    body = _insert_slice_offset_tuple.body
    assert isinstance(body, Call) and isinstance(body.target, InsertSlice)
    offsets = body.args[2]
    assert isinstance(offsets, Tuple), f"offsets is {type(offsets).__name__}, not Tuple"
    assert len(offsets.elements) == 3
    for field in offsets.type.fields:
        assert field.shape == () and field.dtype in (DType.i32, DType.i64)


@dataclass(frozen=True)
class _Cfg:
    head_dim: int = 128
    rms_eps: float = 1e-6


_CFG = _Cfg()
_NK, _KD = 16, 128


@func
def _compile_time_operands(x: Tensor[(1, 2048), "bf16"]) -> Tensor[(1, 16, 128), "bf16"]:
    key_dim = _NK * _KD
    scaled = mul(x, _CFG.head_dim**-0.5)  # noqa: F405
    shifted = add(scaled, _CFG.rms_eps)  # noqa: F405
    return reshape(shifted, new_shape=(1, key_dim // _KD, _KD))  # noqa: F405


def test_compile_time_values_reach_an_op_as_constants() -> None:
    """Test compile time values reach an op as constants.

    ``config.head_dim ** -0.5``, ``config.rms_eps`` and a body-local
    ``_NK * _KD`` are numbers by parse time: the first two arrive as bf16 scalar
    Constants, and the third serves as a shape.
    """
    scaled, eps = _compile_time_operands.body.args[0].args
    operand, scale = scaled.args

    assert isinstance(scale, Constant) and scale.value == pytest.approx(128**-0.5)
    assert isinstance(eps, Constant) and eps.value == pytest.approx(1e-6)
    assert scale.type.dtype == DType.bf16 and eps.type.dtype == DType.bf16
    assert operand.name == "x"
    assert _compile_time_operands.body.target.new_shape == (1, 16, 128)
