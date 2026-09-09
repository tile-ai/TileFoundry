"""The reduce tag table and the Partial attr both name a ``ReduceKind``.

Two guards over the same mapping. ``REDUCE_TAG`` was missing ``MAX`` until a
call site raised ``KeyError`` at emission time, and ``_render_attr`` answered
``shard::P<void>`` for every ``Partial``, so a partial's reduction never
reached C++ at all. Both are one table now, and these pin it.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import pytest

from tilefoundry.codegen.cuda.tir.memory.tensor_view import _render_attr
from tilefoundry.codegen.cuda.tir.reduce import REDUCE_TAG
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.types.shard.shard_layout import Partial


def test_every_reduce_kind_has_a_runtime_tag() -> None:
    """The table covers the enumeration exactly, with nothing dangling.

    The runtime specialises ``reduce_traits`` for five ops and ``ReduceKind``
    names five kinds; a kind with no tag is a ``KeyError`` at emission and a
    tag with no kind is a name nothing can ask for.
    """
    assert set(REDUCE_TAG) == set(ReduceKind)


@pytest.mark.parametrize(
    ("reduction", "tag"),
    [
        ("sum", "add_op"),
        ("mean", "mean_op"),
        ("max", "max_op"),
        ("min", "min_op"),
        ("abs_max", "absmax_op"),
    ],
)
def test_a_partial_carries_its_reduction_into_the_type(reduction: str, tag: str) -> None:
    """``shard::P``'s parameter is the reduction, not ``void``."""
    assert _render_attr(Partial(reduction)) == f"tilefoundry::shard::P<tilefoundry::ops::{tag}>"


def test_a_reduction_the_runtime_cannot_name_is_refused() -> None:
    """An unknown reduction raises rather than degrading to ``P<void>``.

    Falling back is what hid the defect: the type compiled, the attr still read
    as partial, and only the reduction was silently gone.
    """
    with pytest.raises((NotImplementedError, ValueError, KeyError)):
        _render_attr(Partial("median"))
