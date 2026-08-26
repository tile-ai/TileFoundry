"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Mesh, Tensor, Topology
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
