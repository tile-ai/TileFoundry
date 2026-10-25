"""Parser contracts for op calls: operand literals, variadic sequences, and refusals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_origin

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import DType
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.parser import ParseError

FIXTURES = Path(__file__).parents[1] / "fixtures"
DIAGNOSTICS = FIXTURES / "diagnostics"
MATMULS = FIXTURES / "parser" / "matmul_calls.py"
VARIADICS = FIXTURES / "parser" / "variadic_calls.py"
LITERALS = FIXTURES / "parser" / "literal_operands.py"
COMPREHENSIONS = {
    "multiple": "parser_comprehension_two_generators.py",
    "target": "parser_comprehension_tuple_target.py",
    "filter": "parser_comprehension_filter.py",
    "iterator": "parser_comprehension_tuple_iterator.py",
    "bound": "parser_comprehension_tensor_bound.py",
}
REFUSED_MATMULS = {
    "misspelled": "parser_matmul_misspelled_attribute.py",
    "few": "parser_matmul_one_input.py",
    "many": "parser_matmul_six_arguments.py",
    "twice": "parser_matmul_attribute_twice.py",
}


class CallCase(Case):
    """A parsed call checked by the IR it builds."""


def test_matmul_layout_literals_checked() -> None:
    assert get_args(MatMul.a_layout.annotation) == ("MK", "KM")
    assert get_args(MatMul.b_layout.annotation) == ("NK", "KN")

    default_layout = load_authored_ir(f"{MATMULS}:DefaultLayout").entry_function()

    assert isinstance(default_layout.body, Call)
    assert isinstance(default_layout.body.target, MatMul)
    assert (default_layout.body.target.a_layout, default_layout.body.target.b_layout) == (
        "MK",
        "KN",
    )

    with pytest.raises(ParseError, match="b_layout must be one of 'NK', 'KN'"):
        load_authored_ir(str(DIAGNOSTICS / "parser_matmul_bad_layout.py"))


def test_variadic_literals_flatten_to_args() -> None:
    for annotation in (Concat.inputs.annotation, Stack.inputs.annotation):
        assert get_origin(annotation) is tuple
        assert get_args(annotation) == (is_ranked_tensor(),)

    from_list = load_authored_ir(f"{VARIADICS}:ConcatFromList").entry_function()
    from_tuple = load_authored_ir(f"{VARIADICS}:ConcatFromTuple").entry_function()

    for function in (from_list, from_tuple):
        assert isinstance(function.body, Call)
        assert isinstance(function.body.target, Concat)
        assert len(function.body.args) == 2


@case
class StackUsesTheVariadicListContract(CallCase):
    path: Path = VARIADICS
    name: str | None = "StackRows"

    def guard(self, state, problems):
        stack_rows = state.entry_function()

        assert isinstance(stack_rows.body, Call)
        assert isinstance(stack_rows.body.target, Stack)
        assert len(stack_rows.body.args) == 2


def test_empty_variadic_list_reaches_verifier() -> None:
    with pytest.raises(VerifyError, match="Concat requires at least one input"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_empty_list.py"))


@case
class StaticRangeComprehensionInSourceOrder(CallCase):
    path: Path = VARIADICS
    name: str | None = "ComprehensionRows"

    def guard(self, state, problems):
        rows = state.entry_function()

        assert isinstance(rows.body, Call)
        assert isinstance(rows.body.target, Stack)
        assert len(rows.body.args) == 2
        indices = []
        for row in rows.body.args:
            assert isinstance(row, Call) and isinstance(row.target, Reshape)
            sliced = row.args[0]
            assert isinstance(sliced, Call) and isinstance(sliced.target, Slice)
            starts = sliced.args[1]
            assert isinstance(starts, Tuple)
            start = starts.elements[0]
            assert isinstance(start, Constant)
            indices.append(start.value)
        assert indices == [0, 1]


def test_variadic_positional_inputs_rejected() -> None:
    with pytest.raises(ParseError, match="require exactly one list, tuple"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_direct_inputs.py"))

    with pytest.raises(ParseError, match="require exactly one list, tuple"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_positional_axis.py"))


def test_variadic_generator_starred_rejected() -> None:
    with pytest.raises(ParseError, match="do not support generator expressions"):
        load_authored_ir(str(DIAGNOSTICS / "parser_stack_generator.py"))

    with pytest.raises(ParseError, match="do not support starred expansion"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_starred.py"))

    with pytest.raises(ParseError, match="require an explicit list, tuple"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_expanded_name.py"))


def test_variadic_sequence_name_rejected() -> None:
    with pytest.raises(ParseError, match="require an explicit list, tuple"):
        load_authored_ir(str(DIAGNOSTICS / "parser_concat_sequence_name.py"))


@pytest.mark.parametrize(
    "program, message",
    [
        ("multiple", "exactly one generator"),
        ("target", "simple Name target"),
        ("filter", "does not support filters"),
        ("iterator", "range with 1 to 3 static integer arguments"),
        ("bound", "range arguments must be static integers"),
    ],
)
def test_unsupported_comprehension_names_shape(program: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        load_authored_ir(str(DIAGNOSTICS / COMPREHENSIONS[program]))


@case
class PythonFloatLiteralIsF32(CallCase):
    """A literal carries the dtype it is written with; nothing adapts it."""

    path: Path = LITERALS
    name: str | None = "FloatLiteral"

    def guard(self, state, problems):
        add_literal = state.entry_function()

        assert isinstance(add_literal.body, Call)
        constant = next(arg for arg in add_literal.body.args if isinstance(arg, Constant))
        assert constant.type.dtype is DType.f32


def test_mismatched_literal_names_cast() -> None:
    """The remedy is the one the author writes, not one the parser guesses."""
    with pytest.raises(VerifyError, match=re.escape("tf.cast(<operand>, dtype='bf16')")):
        load_authored_ir(str(DIAGNOSTICS / "parser_bf16_float_literal.py"))


@case
class ExplicitCastGivesTheOperandDtype(CallCase):
    path: Path = LITERALS
    name: str | None = "CastLiteral"

    def guard(self, state, problems):
        scale = state.entry_function()

        assert isinstance(scale.body, Call)
        assert scale.body.type.dtype is DType.bf16


def test_python_integer_operand_rejected() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch \\(bf16 vs i64\\)"):
        load_authored_ir(str(DIAGNOSTICS / "parser_bf16_integer_operand.py"))


def test_mixed_dtype_operands_rejected() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch \\(bf16 vs f32\\)"):
        load_authored_ir(str(DIAGNOSTICS / "parser_mixed_dtype_operands.py"))


def test_at_operator_builds_matmul() -> None:
    infix = load_authored_ir(f"{MATMULS}:InfixMatmul").entry_function()
    spelled = load_authored_ir(f"{MATMULS}:SpelledMatmul").entry_function()

    assert isinstance(infix.body, Call)
    assert isinstance(infix.body.target, MatMul)
    assert infix.body.type == spelled.body.type


def test_unresolved_callee_named() -> None:
    with pytest.raises(VerifyError, match="unsupported call 'tf.no_such_op'"):
        load_authored_ir(str(DIAGNOSTICS / "parser_unknown_op.py"))


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (
            "misspelled",
            "matmul has no attribute 'a_layoutt'; its attributes are: a_layout, b_layout, out_dtype",
        ),
        ("few", "matmul takes 2 inputs, got 1"),
        ("many", "matmul takes at most 5 positional arguments, got 6"),
        ("twice", "attribute 'a_layout' is already bound by a positional argument"),
    ],
)
def test_refused_call_states_reason(program: str, message: str) -> None:
    """A call the parser recognized and refused names its own reason, not a shape mismatch."""
    with pytest.raises(ParseError, match=re.escape(message)):
        load_authored_ir(str(DIAGNOSTICS / REFUSED_MATMULS[program]))


def test_refused_call_reports_one_reason() -> None:
    """A choice drops the alternatives that stated nothing instead of listing them."""
    with pytest.raises(ParseError) as raised:
        load_authored_ir(str(DIAGNOSTICS / "parser_matmul_one_input.py"))

    message = str(raised.value)
    assert "pattern did not match" not in message
    assert "no alternative matched" not in message
    assert message.count("matmul takes 2 inputs, got 1") == 1


def test_unclaimed_call_named_last() -> None:
    """Position is the whole claim for the report of last resort.

    The report runs only after every alternative has declined, so it names any
    callee — inside the op namespace or not — while a call another pattern owns
    never reaches it. ``launch(...)`` is such a call and stays parseable.
    """
    with pytest.raises(ParseError, match=re.escape("unsupported call 'no_such_helper'")):
        load_authored_ir(str(DIAGNOSTICS / "parser_unclaimed_call.py"))

    host = load_authored_ir(f"{FIXTURES}/parser/launch_call.py:LaunchHost").entry_function()

    assert host.body is not None


def test_function_mesh_requires_declared_topology() -> None:
    with pytest.raises(ParseError, match="topology 'cta' not declared by @module"):
        load_authored_ir(str(DIAGNOSTICS / "parser_undeclared_topology.py"))


CASES = special(CallCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_call_case(case: CallCase) -> None:
    check_case(case)
