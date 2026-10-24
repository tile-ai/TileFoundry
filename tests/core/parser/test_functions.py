"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.core.parser._fixtures import check_parsed_call_sources
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.parser import ParseError


def test_a_function_may_have_a_leading_docstring() -> None:
    @func
    def documented(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        """This is documentation, not an HIR statement."""
        return x

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
        if iterator == "tf.tile(10)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tf.tile(10):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out

        elif iterator == "tf.tile(1, 2, 3, 4)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tf.tile(1, 2, 3, 4):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out

        elif iterator == "range(1, 2, 3, 4)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in range(1, 2, 3, 4):
                    out = tf.add(x[row, :], seed)
                return out

        else:

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in steps(1, 2):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out


_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
_FIXTURE_SOURCES = tuple(
    path for path in sorted(_FIXTURE_ROOT.rglob("*.py")) if path.name != "__init__.py"
)


@pytest.mark.parametrize(
    "source", _FIXTURE_SOURCES, ids=lambda source: str(source.relative_to(_FIXTURE_ROOT))
)
def test_every_parsed_call_knows_where_it_came_from(source: Path) -> None:
    check_parsed_call_sources(source)
