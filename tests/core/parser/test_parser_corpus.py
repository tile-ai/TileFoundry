"""Core inspection programs parsed and printed once per input."""

from pathlib import Path

import pytest

from tests.core._corpus import cases
from tests.core.parser._parser_case import ParserCase, check_parser_case, unique_parser_cases
from tests.core.parser._parser_hooks import (
    _carry_order,
    _dtype_names,
    _nan_attr,
    _nan_fill,
    _nan_tuple,
    _nested_scan,
    _scalar_range,
    _sequential_scan,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
EXPECTED = {
    "inspection/f_positional": dict(same_as=FIXTURES / "inspection/f_keyword.py"),
    "inspection/cut": dict(prints=("tf.slice(x, (start, 0), sizes=(2, 4), strides=(1, 1))",)),
    "inspection/gather_unit": dict(prints=("for i in range(8):",), hooks=(_scalar_range,)),
    "inspection/gather_stride": dict(prints=("for i in range(0, 8, 2):",), hooks=(_scalar_range,)),
    "inspection/sh": dict(prints=("topk = tf.topk(",)),
    "inspection/lp": dict(prints=("fp8e4m3", "f8e8m0", "f4e2m1"), hooks=(_dtype_names,)),
    "inspection/f_mesh_tuple": dict(prints=("return (", "mesh=Mesh(")),
    "inspection/f_nested_layout": dict(prints=("layout=ComposedLayout(", "outer=ShardLayout(")),
    "inspection/online": dict(hooks=(_carry_order,)),
    "inspection/mixed": dict(prints=("@func", "@prim_func")),
    "inspection/control": dict(prints=("for i in range(0, 2, 1):", "if i < 1:")),
    "inspection/nan_value_tuple": dict(prints=("math.nan",), hooks=(_nan_tuple,)),
    "inspection/nan_value_attr": dict(prints=("math.nan",), hooks=(_nan_attr,)),
    "inspection/nan_value_fill": dict(prints=("math.nan",), hooks=(_nan_fill,)),
}
TILE_WINDOWS = FIXTURES / "shapes/tile_window_syntax.py"
NAMED = (
    ParserCase(
        path=TILE_WINDOWS, corpus=TILE_WINDOWS.parent, name="scan_copy", prints=("x[row, :]",)
    ),
    ParserCase(
        path=TILE_WINDOWS,
        corpus=TILE_WINDOWS.parent,
        name="nested_scan_copy",
        prints=("x[row, col]",),
        hooks=(_nested_scan,),
    ),
    ParserCase(
        path=TILE_WINDOWS,
        corpus=TILE_WINDOWS.parent,
        name="sequential_scan_copy",
        prints=("x[row, :]",),
        hooks=(_sequential_scan,),
    ),
)
CASES = cases(FIXTURES / "inspection", case_type=ParserCase, expected=EXPECTED) + NAMED
CASES = unique_parser_cases(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
