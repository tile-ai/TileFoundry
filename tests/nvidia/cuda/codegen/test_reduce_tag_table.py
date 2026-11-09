"""The reduce tag table and the Partial attr both name a ``ReduceKind``.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir import expr, types
from tilefoundry_nvidia.cuda.codegen import tir
from tilefoundry_nvidia.cuda.codegen.tir import memory


def test_every_reduce_kind_has_a_runtime_tag() -> None:
    """The table covers the enumeration exactly, with nothing dangling."""
    assert set(tir.REDUCE_TAG) == set(expr.ReduceKind)


@pytest.mark.parametrize(
    ("reduction", "tag"),
    [
        ("sum", "primitive::add_op"),
        ("mean", "ops::mean_op"),
        ("max", "primitive::max_op"),
        ("min", "primitive::min_op"),
        ("abs_max", "ops::absmax_op"),
    ],
)
def test_a_partial_carries_its_reduction_into_the_type(reduction: str, tag: str) -> None:
    """``shard::P``'s parameter is the reduction, not ``void``.

    Which namespace it comes from says what kind of thing it is: a callable an
    op is handed lives in ``primitive``, while ``mean`` and ``abs_max`` are
    reduce's own tags and live with it.
    """
    assert memory.render_attr(types.Partial(reduction)) == f"tilefoundry::shard::P<tilefoundry::{tag}>"


def test_a_reduction_the_runtime_cannot_name_is_refused() -> None:
    """An unknown reduction raises rather than degrading to ``P<void>``."""
    with pytest.raises((NotImplementedError, ValueError, KeyError)):
        memory.render_attr(types.Partial("median"))
