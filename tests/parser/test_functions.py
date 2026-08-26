"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re

import pytest

from tilefoundry import func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.inspection import as_script
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget


def test_a_function_may_have_a_leading_docstring() -> None:
    @func
    def documented(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        """This is documentation, not an HIR statement."""
        return x

    assert "This is documentation" not in as_script(documented)


def test_a_nested_block_does_not_gain_function_docstring_semantics() -> None:
    with pytest.raises(ParseError):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def nested(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("unit",)) as _mesh:
                """A string in a with body remains an ordinary statement."""
                return x


@pytest.mark.parametrize(
    ("iterator", "expected"),
    (
        ("tile(10)", "tile(extent) is not supported; use range(extent)"),
        ("tile(1, 2, 3)", "tile() takes 2 arguments (extent, step), got 3"),
        ("range(1, 2, 3, 4)", "range() takes 1 to 3 arguments, got 4"),
        ("steps(1, 2)", "loop iterator must be tile(...) or range(...)"),
    ),
)
def test_a_loop_iterator_states_why_its_arity_is_invalid(iterator: str, expected: str) -> None:
    """The shape-exact syntax must not reduce these to a bare match failure.

    Encoding each iterator's arity is what lets the generated grammar show the
    accepted forms, so the reason is stated before the shape rejects.
    """
    with pytest.raises(ParseError, match=re.escape(expected)):
        if iterator == "tile(10)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tile(10):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out

        elif iterator == "tile(1, 2, 3)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tile(1, 2, 3):  # noqa: F821
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
