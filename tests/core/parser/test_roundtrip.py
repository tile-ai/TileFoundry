"""Parser-owned checks for canonical grid-loop bindings."""


from pathlib import Path

import pytest

from tilefoundry import func
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.specialize import InstantiateDimensionCloner


def test_gqa_correction_reads_the_old_carry_and_unique_yield() -> None:
    gqa_decode = Path(__file__).parents[1] / "fixtures" / "placed" / "gqa_decode.py"
    entry = load_authored_ir(f"{gqa_decode}:GqaOnline").entry_function()
    function = InstantiateDimensionCloner({"ctx_len": 8}).function(entry)
    printed = as_script(function)

    assert printed.count("        m_new = tf.max(m, score)") == 1
    assert " = tf.sub(m, m_new)" in printed
    lines = printed.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.lstrip() == "for i in range(8):"
    )
    loop_indent = lines[start][: -len(lines[start].lstrip())]
    body_indent = loop_indent + "    "
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].startswith(loop_indent)
        and not lines[index].startswith(body_indent)
    )
    yielded = lines[end - 3 : end]
    assert [line.split(" = ", 1)[0] for line in yielded] == [
        f"{body_indent}l",
        f"{body_indent}o",
        f"{body_indent}m",
    ]
    assert yielded[-1] == f"{body_indent}m = m_new"
    assert len({line.split(" = ", 1)[1] for line in yielded}) == 3
    assert lines[end] == f'{loop_indent}k_n = tf.cast(k_new, dtype="f32")'


def negative_float_literal(x: Tensor[(), "f32"]):
    return -1.0


def negative_integer_literal(x: Tensor[(), "f32"]):
    return -3


def negative_infinity_literal(x: Tensor[(), "f32"]):
    return -1e999


def negated_value(x: Tensor[(), "f32"]):
    return -x


@pytest.mark.parametrize(
    ("program", "expected"),
    (
        pytest.param(negative_float_literal, -1.0, id="negative_float_literal"),
        pytest.param(negative_integer_literal, -3, id="negative_integer_literal"),
        pytest.param(negative_infinity_literal, float("-inf"), id="negative_infinity_literal"),
        pytest.param(negated_value, None, id="negated_value"),
    ),
)
def test_signed_literals_are_constants_and_negated_values_are_calls(program, expected) -> None:
    """Literal signs belong to constants; negating a runtime value remains a call."""
    function = func(program)
    result = function.body
    if expected is None:
        assert isinstance(result, Call)
        assert isinstance(result.target, Unary)
        assert result.target.kind is UnaryKind.NEG
        assert result.args[0] is function.params[0]
    else:
        assert isinstance(result, Constant)
        assert type(result.value) is type(expected)
        assert result.value == expected
