"""Core inspection programs parsed and printed once per input."""

import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.core._corpus import case, cases
from tests.core.parser._parser_case import ParserCase, check_parser_case, unique_parser_cases
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Constant
from tilefoundry.ir.types import DType

FIXTURES = Path(__file__).parents[1] / "fixtures"
TILE_WINDOWS = FIXTURES / "shapes/tile_window_syntax.py"


@dataclass(frozen=True)
class ScalarRange(ParserCase):
    def guard(self, state, problems):
        assert "for i in tf.tile(" not in state.printed


@case
class GatherUnit(ScalarRange):
    path: Path = FIXTURES / "inspection/gather_unit.py"
    prints: tuple[str, ...] = ("for i in range(8):",)


@case
class GatherStride(ScalarRange):
    path: Path = FIXTURES / "inspection/gather_stride.py"
    prints: tuple[str, ...] = ("for i in range(0, 8, 2):",)


@case
class Lp(ParserCase):
    path: Path = FIXTURES / "inspection/lp.py"
    prints: tuple[str, ...] = ("fp8e4m3", "f8e8m0", "f4e2m1")

    def guard(self, state, problems):
        ir, rebuilt = state.ir, state.rebuilt
        expected = [DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1]
        assert [p.type.dtype for p in ir.entry_function().params] == expected
        assert [p.type.dtype for p in rebuilt.entry_function().params] == expected


@case
class Online(ParserCase):
    path: Path = FIXTURES / "inspection/online.py"

    def guard(self, state, problems):
        ir, printed, rebuilt = state.ir, state.printed, state.rebuilt
        loop_lines = printed[printed.index("        for i in range(4):") :].splitlines()

        assert loop_lines[-3:] == [
            "            o = o_2",
            "            m = m_new",
            "        return (m, o)",
        ]
        assert loop_lines.index("            m_new = tf.max(m, x)") < loop_lines.index(
            "            corr = tf.sub(m, m_new)"
        )
        assert repr(ir.entry_function().body) == repr(rebuilt.entry_function().body)


@case
class NanValueTuple(ParserCase):
    path: Path = FIXTURES / "inspection/nan_value_tuple.py"
    prints: tuple[str, ...] = ("math.nan",)

    def guard(self, state, problems):
        body = state.rebuilt.entry_function().body
        assert isinstance(body.elements[1], Constant)
        assert math.isnan(body.elements[1].value)


@case
class NanValueAttr(ParserCase):
    path: Path = FIXTURES / "inspection/nan_value_attr.py"
    prints: tuple[str, ...] = ("math.nan",)

    def guard(self, state, problems):
        assert math.isnan(state.rebuilt.entry_function().body.target.value)


@case
class NestedScanCopy(ParserCase):
    path: Path = TILE_WINDOWS
    name: str | None = "NestedScanCopy"
    prints: tuple[str, ...] = ("x[row, col]",)

    def guard(self, state, problems):
        lines = state.printed.splitlines()
        inner = next(
            i for i, line in enumerate(lines) if line.strip() == "for col in tf.tile(4, 2):"
        )
        assert lines[inner - 1].strip() != "out = out"
        assert lines[-2].strip() == "out = out"


@case
class SequentialScanCopy(ParserCase):
    path: Path = TILE_WINDOWS
    name: str | None = "SequentialScanCopy"
    prints: tuple[str, ...] = ("x[row, :]",)

    def guard(self, state, problems):
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        torch.testing.assert_close(evaluate(state.rebuilt.entry_function(), x), (x + 1) * 2)


EXPECTED = {
    "inspection/f_positional": dict(same_as=FIXTURES / "inspection/f_keyword.py"),
    "inspection/cut": dict(prints=("tf.slice(x, (start, 0), sizes=(2, 4), strides=(1, 1))",)),
    "inspection/sh": dict(prints=("topk = tf.topk(",)),
    "inspection/f_mesh_tuple": dict(prints=("return (", "mesh=Mesh(")),
    "inspection/f_nested_layout": dict(prints=("layout=ComposedLayout(", "outer=ShardLayout(")),
    "inspection/mixed": dict(prints=("@func", "@prim_func")),
    "inspection/control": dict(prints=("for i in range(0, 2, 1):", "if i < 1:")),
}
NAMED = (
    ParserCase(path=TILE_WINDOWS, name="ScanCopy", prints=("x[row, :]",)),
    NestedScanCopy(),
    SequentialScanCopy(),
)
CASES = cases(FIXTURES / "inspection", case_type=ParserCase, expected=EXPECTED) + NAMED
CASES = unique_parser_cases(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
