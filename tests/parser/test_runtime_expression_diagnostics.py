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


@pytest.mark.parametrize(
    ("iterator", "expected"),
    (
        ("tile(10)", "tile(extent) is not supported; use range(extent)"),
        ("tile(1, 2, 3)", "tile() takes 2 arguments (extent, step), got 3"),
        ("range(1, 2, 3, 4)", "range() takes 1 to 3 arguments, got 4"),
        ("steps(1, 2)", "loop iterator must be tile(...) or range(...)"),
    ),
)
def test_loop_iterator_arity_states_its_own_reason(iterator: str, expected: str) -> None:
    """The shape-exact syntax must not degrade these into a bare match failure.

    Encoding each iterator's arity is what lets the generated grammar show the
    accepted forms, so the specific reason is stated before the shape rejects.
    """
    with pytest.raises(VerifyError) as caught:
        import_dsl(
            _HEADER
            + f'''@func
def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
    out = tf.add(seed, seed)
    for row in {iterator}:
        out = tf.add(x[row, :], seed)
    return out
'''
        )

    assert str(caught.value).splitlines()[0].startswith(f"loop_header: {expected}")
