"""Diagnostics for unsupported authored runtime expressions."""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry.ir.core import VerifyError

_HEADER = """from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

"""


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        (
            "return tf.grouped_matmul(x, x)",
            "runtime_expression: unsupported call 'tf.grouped_matmul' "
            "(2 positional, no keywords)",
        ),
        (
            "return {'value': x}",
            "runtime_expression: unsupported Dict syntax \"{'value': x}\"",
        ),
        (
            "return tf.grouped_matmul(x, x) + x",
            "runtime_expression: unsupported call 'tf.grouped_matmul' "
            "(2 positional, no keywords)",
        ),
    ),
)
def test_unsupported_expression_names_its_shape(body: str, expected: str) -> None:
    with pytest.raises(VerifyError) as caught:
        import_dsl(
            _HEADER
            + f"""@func
def unsupported(x: Tensor[(8,), "bf16"]):
    {body}
"""
        )

    assert str(caught.value).splitlines()[0] == expected
