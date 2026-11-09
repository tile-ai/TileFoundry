"""Verify ``tir.ShapeOf`` construction."""

from __future__ import annotations

import pytest

from tilefoundry import verifier
from tilefoundry.ir import core, expr, types


def _x_param() -> expr.Var:
    return expr.Var(
        type=types.TensorType(shape=(4, 8), dtype=expr.DType.f32, layout=None, storage="gmem"), name="x"
    )


def test_shape_of_rejects_negative_axis():
    p = _x_param()
    so = core.ShapeOf(
        type=types.TensorType.from_scalar(dtype=expr.DType.i32, storage=expr.StorageKind.RMEM),
        param=p,
        axis=-1,
    )
    with pytest.raises(expr.VerifyError, match="non-negative"):
        verifier.verify_shape_of(so)


def test_shape_of_rejects_wrong_type():
    p = _x_param()
    so = core.ShapeOf(
        type=types.TensorType(shape=(4,), dtype=expr.DType.i32, layout=None, storage="rmem"),
        param=p,
        axis=0,
    )
    with pytest.raises(expr.VerifyError, match="rank-0 i32"):
        verifier.verify_shape_of(so)
