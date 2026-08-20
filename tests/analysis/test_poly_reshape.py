"""Pin ``extract``'s zero-op view fold for ``Reshape``.

Reshape contributes no statement or buffer: consumers resolve through to the
source and fold the coordinates through the Op's own registered access
relation, which is the only place the renaming is stated. Distinct toy
dimensions mirror decoder head split/merge shapes while keeping exact
per-element maps readable.
"""

from __future__ import annotations

import isl
import pytest

from tilefoundry import func
from tilefoundry.analysis import ExtractError, TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- reshape/sigmoid resolved dynamically

B, S, H, D = 1, 4, 3, 8
HD = H * D


@func
def split_then_sigmoid(x: Tensor[(B, S, HD), "f32"]) -> Tensor[(B, S, H, D), "f32"]:
    y = reshape(x, new_shape=(B, S, H, D))
    z = sigmoid(y)
    return z


@func
def merge_then_sigmoid(v: Tensor[(B, S, H, D), "f32"]) -> Tensor[(B, S, HD), "f32"]:
    w = reshape(v, new_shape=(B, S, HD))
    u = sigmoid(w)
    return u


@func
def bare_reshape_return(x4: Tensor[(B, S, HD), "f32"]) -> Tensor[(B, S, H, D), "f32"]:
    y5 = reshape(x4, new_shape=(B, S, H, D))
    return y5


def test_a_reshape_is_a_view_its_consumer_reads_through():
    """Both directions, because they are not the same construction.

    Splitting a merged head axis composes the consumer read through to the source
    with ``D*h + d``. Merging reconstructs ``(h, d)`` with div/mod. In both
    directions reshape contributes no statement and its SSA name is not a buffer.
    """
    split = extract(split_then_sigmoid)
    assert isinstance(split, TileGraph)
    assert [type(u.op.target).__name__ for u in split.units] == ["Sigmoid"]
    bounds = f"0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D}"
    assert split.domain.is_equal(isl.union_set(f"{{ Sigmoid[b,s,h,d] : {bounds} }}"))
    assert split.reads.is_equal(
        isl.union_map(f"{{ Sigmoid[b,s,h,d] -> x[b,s,{D}*h+d] : {bounds} }}")
    )
    assert split.writes.is_equal(isl.union_map(f"{{ Sigmoid[b,s,h,d] -> z[b,s,h,d] : {bounds} }}"))
    assert "y[" not in str(split.reads) and "y[" not in str(split.writes)
    assert split.deps.is_empty()

    merged = extract(merge_then_sigmoid)
    assert [type(u.op.target).__name__ for u in merged.units] == ["Sigmoid"]
    assert merged.reads.is_equal(
        isl.union_map(
            f"{{ Sigmoid[b,s,e] -> v[b,s,floor(e/{D}),e mod {D}] : "
            f"0<=b<{B} and 0<=s<{S} and 0<=e<{HD} }}"
        )
    )
    assert "w[" not in str(merged.reads)


def test_boundary_reshape_with_no_consumer_fails_closed():
    """Test boundary reshape with no consumer fails closed.

    A body that is *nothing but* a reshape (``return reshape(x, ...)``) has no
    compute op left once the reshape folds away -- ``extract`` fails closed with
    its empty-body error rather than fabricating a copy statement.
    """
    with pytest.raises(ExtractError, match="no compute ops to extract"):
        extract(bare_reshape_return)
