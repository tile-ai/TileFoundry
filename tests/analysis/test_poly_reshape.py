"""``extract`` coverage for ``Reshape`` -- the view-fold (not a fallback
relation like ``RMSNorm``, not a registered ``type_relation`` like
``Transpose``): before this task ``Reshape`` had no forward ``type_relation``
and no V1 fallback, so ``analysis.extract`` raised at any ``Reshape`` call
(the last blocker to a real decoder layer's whole self-attention/mlp
extracting -- a decoder reshapes q/k/v into heads and reshapes ``attn_out``
back).

A reshape is a zero-op at the buffer level (same memory, reinterpreted), so
``extract`` never gives it a statement: ``extract()``'s postorder walk skips
it exactly like ``TupleGetItem``, and ``_buffer_namer`` resolves any reference
to its output straight through to its source buffer's name. Unlike
``TupleGetItem`` (a pure name passthrough), a consumer's *access map* also
needs recomposing -- the source has a different coordinate space -- via
``namer.pierce``, which composes the consumer's own read formula with
``reshape.flat_reshape_map`` (row-major flat-index equality between old/new
shape, isl div/mod for a merge, plain multiply-add for a split; see that
function's docstring for the general construction). Shapes below (``H=3, D=8,
HD=24``, all pairwise distinct from ``B=1, S=4``) mirror a decoder's
head-split/merge reshape at toy size; extraction is plain element granularity
throughout, so the asserted access maps are exact per-element formulas.
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

    Splitting a merged head axis (the q/k/v-projection shape): the consumer's
    read is not of the reshape's output at all, it is composed straight through
    to the source's own merged coordinates, and the new (h, d) pair reconstructs
    the old flat offset by plain multiply-add (``D*h + d``) -- no div/mod needed.

    Merging back (the ``attn_out`` shape before the output projection):
    reconstructing the *old* split (h, d) from the *new* flat offset genuinely
    needs div/mod. That is inherent to unpacking a flat index rather than a design
    gap. Either way the reshape contributes no statement and its own SSA name
    never becomes a buffer, which is what makes the fold observable.
    """
    split = extract(split_then_sigmoid)
    assert isinstance(split, TileGraph)
    assert [type(u.op.target).__name__ for u in split.units] == ["Sigmoid"]
    bounds = f"0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D}"
    assert split.domain.is_equal(isl.union_set(f"{{ Sigmoid[b,s,h,d] : {bounds} }}"))
    assert split.reads.is_equal(
        isl.union_map(f"{{ Sigmoid[b,s,h,d] -> x[b,s,{D}*h+d] : {bounds} }}")
    )
    assert split.writes.is_equal(
        isl.union_map(f"{{ Sigmoid[b,s,h,d] -> z[b,s,h,d] : {bounds} }}")
    )
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
    """A body that is *nothing but* a reshape (``return reshape(x, ...)``) has no
    compute op left once the reshape folds away -- ``extract`` fails closed with
    its empty-body error rather than fabricating a copy statement."""
    with pytest.raises(ExtractError, match="no compute ops to extract"):
        extract(bare_reshape_return)
