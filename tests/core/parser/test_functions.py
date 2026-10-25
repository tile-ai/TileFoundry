"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.parser import ParseError

FIXTURES = Path(__file__).parents[1] / "fixtures"
LOOP_ITERATORS = {
    "tf.tile(10)": "parser_tile_one_argument.py",
    "tf.tile(1, 2, 3, 4)": "parser_tile_four_arguments.py",
    "range(1, 2, 3, 4)": "parser_range_four_arguments.py",
    "steps(1, 2)": "parser_unknown_loop_iterator.py",
}


def test_a_function_may_have_a_leading_docstring() -> None:
    documented = load_authored_ir(
        f"{FIXTURES}/parser/documented_function.py:Documented"
    ).entry_function()

    assert "This is documentation" not in as_script(documented)


@pytest.mark.parametrize(
    ("iterator", "expected"),
    (
        ("tf.tile(10)", "tf.tile(extent) is not supported; use range(extent)"),
        (
            "tf.tile(1, 2, 3, 4)",
            "tf.tile() takes 2 or 3 arguments, (stop, step) or (start, stop, step), got 4",
        ),
        ("range(1, 2, 3, 4)", "range() takes 1 to 3 arguments, got 4"),
        ("steps(1, 2)", "loop iterator must be tf.tile(...) or range(...)"),
    ),
)
def test_a_loop_iterator_states_why_its_arity_is_invalid(iterator: str, expected: str) -> None:
    """The shape-exact syntax must not reduce these to a bare match failure.

    Encoding each iterator's arity is what lets the generated grammar show the
    accepted forms, so the reason is stated before the shape rejects.
    """
    with pytest.raises(ParseError, match=re.escape(expected)):
        load_authored_ir(str(FIXTURES / "diagnostics" / LOOP_ITERATORS[iterator]))
