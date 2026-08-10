"""Verify ``tir.ShapeOf`` construction + ``shape_var_name`` helper."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.verify import _verify_shape_of
from tilefoundry.ir.types import DType, TensorType


def _x_param() -> Var:
    return Var(
        type=TensorType(shape=(4, 8), dtype=DType.f32, layout=None, storage="gmem"), name="x"
    )


def test_shape_of_rejects_negative_axis():
    p = _x_param()
    so = ShapeOf(type=TensorType.scalar(dtype=DType.i32), param=p, axis=-1)
    with pytest.raises(VerifyError, match="non-negative"):
        _verify_shape_of(so)


def test_shape_of_rejects_wrong_type():
    p = _x_param()
    so = ShapeOf(
        type=TensorType(shape=(4,), dtype=DType.i32, layout=None, storage="rmem"),
        param=p,
        axis=0,
    )
    with pytest.raises(VerifyError, match="rank-0 i32"):
        _verify_shape_of(so)
