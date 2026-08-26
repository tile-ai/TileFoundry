"""Parser rules for weak Python float operands."""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry.ir.core import Call, Constant, VerifyError
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType

_HEADER = """from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

"""


def _constant_argument(source: str) -> Constant:
    function = import_dsl(_HEADER + source)
    assert isinstance(function.body, Call)
    assert isinstance(function.body.target, Binary)
    constant = next(arg for arg in function.body.args if isinstance(arg, Constant))
    return constant


def test_float_literal_adopts_the_tensor_dtype() -> None:
    constant = _constant_argument(
        """@func
def add_literal(x: Tensor[(8,), "bf16"]):
    return x + 1.0
"""
    )

    assert constant.type.dtype is DType.bf16


def test_float_closure_adopts_the_tensor_dtype() -> None:
    constant = _constant_argument(
        """alpha = 1.0

@func
def add_closure(x: Tensor[(8,), "bf16"]):
    return tf.add(x, alpha)
"""
    )

    assert constant.type.dtype is DType.bf16


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("x * -0.5", -0.5),
        ("x / -2.0", -2.0),
        ("x + (-1.0)", -1.0),
    ),
)
def test_negative_float_stays_weak_through_unary_minus(expression: str, expected: float) -> None:
    constant = _constant_argument(
        f"""@func
def negative_float(x: Tensor[(8,), "bf16"]):
    return {expression}
"""
    )

    assert constant.type.dtype is DType.bf16
    assert constant.value == expected


def test_integer_literal_remains_i64_and_names_the_float_fix() -> None:
    with pytest.raises(VerifyError, match=r"bf16 vs i64.*write 1\.0"):
        import_dsl(
            _HEADER
            + """@func
def add_integer(x: Tensor[(8,), "bf16"]):
    return x + 1
"""
        )


def test_tensor_operands_do_not_promote() -> None:
    with pytest.raises(VerifyError, match=r"bf16 vs f32.*never promoted"):
        import_dsl(
            _HEADER
            + """@func
def add_tensors(x: Tensor[(8,), "bf16"], y: Tensor[(8,), "f32"]):
    return x + y
"""
        )
