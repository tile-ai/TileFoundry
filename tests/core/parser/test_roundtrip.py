"""Parser-owned checks for canonical grid-loop bindings."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.specialize import InstantiateDimensionCloner

FIXTURES = Path(__file__).parents[1] / "fixtures"
GQA_DECODE = FIXTURES / "placed" / "gqa_decode.py"
SIGNED_LITERALS = FIXTURES / "parser" / "signed_literals.py"


class RoundtripCase(Case):
    """A parsed program checked by the IR it builds or the script it prints."""


@case
class GqaCorrectionReadsOldCarryAndUniqueYield(RoundtripCase):
    path: Path = GQA_DECODE

    def guard(self, state, problems):
        entry = state.entry_function()
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
        assert isinstance(result, Constant)
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
        assert isinstance(result, Call)
        assert isinstance(result.target, Unary)
        assert result.target.kind is UnaryKind.NEG
        assert result.args[0] is function.params[0]


CASES = special(RoundtripCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_roundtrip_case(case: RoundtripCase) -> None:
    check_case(case)
