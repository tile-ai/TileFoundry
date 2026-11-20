"""Parser-owned checks for canonical grid-loop bindings."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import inspection
from tilefoundry.interfaces import valuation
from tilefoundry.ir import expr
from tilefoundry.ir.hir import math

FIXTURES = Path(__file__).parents[1] / "fixtures"
GQA_DECODE = FIXTURES / "placed" / "gqa_decode.py"
SIGNED_LITERALS = FIXTURES / "parser" / "signed_literals.py"
SLICE_LAYOUTS = FIXTURES / "parser" / "slice_layouts.py"


class RoundtripCase(Case):
    """A parsed program checked by the IR it builds or the script it prints."""


@case
class GqaCorrectionReadsOldCarryAndUniqueYield(RoundtripCase):
    path: Path = GQA_DECODE

    def guard(self, state, problems):
        entry = state.entry_function()
        function = valuation.InstantiateDimensionCloner({"ctx_len": 8}).function(entry)
        printed = inspection.as_script(function)

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
            if lines[index].startswith(loop_indent) and not lines[index].startswith(body_indent)
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


@dataclass(frozen=True)
class SignedLiteral(RoundtripCase):
    """A literal's sign belongs to its constant."""

    expected: float | int | None = None

    def guard(self, state, problems):
        result = state.entry_function().body
        assert isinstance(result, expr.Constant)
        assert type(result.value) is type(self.expected)
        assert result.value == self.expected


@case
class NegativeFloatLiteral(SignedLiteral):
    path: Path = SIGNED_LITERALS
    name: str | None = "NegativeFloatLiteral"
    expected: float | int | None = -1.0


@case
class NegativeIntegerLiteral(SignedLiteral):
    path: Path = SIGNED_LITERALS
    name: str | None = "NegativeIntegerLiteral"
    expected: float | int | None = -3


@case
class NegativeInfinityLiteral(SignedLiteral):
    path: Path = SIGNED_LITERALS
    name: str | None = "NegativeInfinityLiteral"
    expected: float | int | None = float("-inf")


@case
class NegatedValue(RoundtripCase):
    """Negating a runtime value remains a call."""

    path: Path = SIGNED_LITERALS
    name: str | None = "NegatedValue"

    def guard(self, state, problems):
        function = state.entry_function()
        result = function.body
        assert isinstance(result, expr.Call)
        assert isinstance(result.target, math.Unary)
        assert result.target.kind is expr.UnaryKind.NEG
        assert result.args[0] is function.params[0]


@case
class SliceLayouts(RoundtripCase):
    """A type on a slice prints as the slice it names and reparses to the same type."""

    path: Path = SLICE_LAYOUTS

    def guard(self, state, problems):
        printed = inspection.as_script(state)
        assert "ShardLayout(" not in printed
        for written in (
            '(4, 8) @ gpu[:1]]',
            '(4, 8) @ {gpu[:1].pp @ V()}]',
            '@ gpu[1:].cp',
            'layout=(4, 8) @ gpu[1:]',
            '((8, 8) | Swizzle(1, 2, 3)) @ gpu[1:]]',
        ):
            assert written in printed
        again = reload(printed, state.name)
        assert inspection.as_script(again) == printed
        assert [param.type for param in again.entry_function().params] == [
            param.type for param in state.entry_function().params
        ]


CASES = special(RoundtripCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_roundtrip_case(case: RoundtripCase) -> None:
    check_case(case)
