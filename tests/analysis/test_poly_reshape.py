"""``extract`` coverage for ``Reshape`` -- the view-fold (not a fallback
relation like ``RMSNorm``, not a registered ``type_relation`` like
``Transpose``): before this task ``Reshape`` had no forward ``type_relation``
and no V1 fallback, so ``analysis.extract`` raised at any ``Reshape`` call
(the last blocker to a real decoder layer's whole self-attention/mlp
extracting -- ``tests/models/qwen3_1_7b/model/decoder_layer.py`` reshapes q/k/v
into heads and reshapes ``attn_out`` back).

A reshape is a zero-op at the buffer level (same memory, reinterpreted), so
``extract`` never gives it a statement: ``extract()``'s postorder walk skips
it exactly like ``TupleGetItem`` (see the module docstring), and
``_buffer_namer`` resolves any reference to its output straight through to
its source buffer's name. Unlike ``TupleGetItem`` (a pure name passthrough),
a consumer's *access map* also needs recomposing -- the source has a
different coordinate space -- via ``namer.pierce``, which composes the
consumer's own read formula with ``reshape.flat_reshape_map`` (row-major
flat-index equality between old/new shape, isl div/mod for a merge, plain
multiply-add for a split; see that function's docstring for the general
construction). Shapes below (``H=3, D=8, HD=24``, all pairwise distinct from
``B=1, S=4``) mirror the qwen3 head-split/merge reshape shape the task calls
out, at toy size; extraction is plain element granularity throughout, so the
asserted access maps are exact per-element formulas.
"""
from __future__ import annotations

import isl
import pytest

from tilefoundry import func
from tilefoundry.analysis import ExtractError, TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- reshape/sigmoid/exp/add resolved dynamically

B, S, H, D = 1, 4, 3, 8
HD = H * D


@func
def split_then_sigmoid(x: Tensor[(B, S, HD), "f32"]) -> Tensor[(B, S, H, D), "f32"]:
    y = reshape(x, new_shape=(B, S, H, D))
    z = sigmoid(y)
    return z


def test_reshape_split_is_not_a_statement():
    """``y = reshape(x, [B,S,H,D])`` (splitting a merged head axis, the
    qwen3 q/k/v-projection shape) contributes zero units -- only
    ``Sigmoid`` extracts."""
    tg = extract(split_then_sigmoid)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    assert tg.units[0].name == "Sigmoid"
    assert type(tg.units[0].op.target).__name__ == "Sigmoid"
    assert "Reshape" not in [type(u.op.target).__name__ for u in tg.units]


def test_reshape_split_pierces_to_the_real_buffer():
    """Sigmoid's read is not of ``y`` at all -- it is composed straight
    through the reshape to ``x``'s own (merged-axis) coordinates: no
    div/mod needed for a split, the new (h,d) pair reconstructs the old
    flat offset by plain multiply-add (``D*h + d``)."""
    tg = extract(split_then_sigmoid)

    print("\n=== reshape split: domain ===")
    print(tg.domain)
    print("=== reshape split: reads ===")
    print(tg.reads)
    print("=== reshape split: writes ===")
    print(tg.writes)

    assert tg.domain.is_equal(
        isl.union_set(f"{{ Sigmoid[b,s,h,d] : 0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} }}")
    )
    expected_reads = isl.map(
        f"{{ Sigmoid[b,s,h,d] -> x[b,s,{D}*h+d] : 0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} }}"
    )
    expected_writes = isl.map(
        f"{{ Sigmoid[b,s,h,d] -> z[b,s,h,d] : 0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} }}"
    )
    assert tg.reads.is_equal(expected_reads)
    assert tg.writes.is_equal(expected_writes)
    # `y` (the reshape's own SSA name) never appears anywhere -- it never
    # got a buffer of its own.
    assert "y[" not in str(tg.reads) and "y[" not in str(tg.writes)
    assert tg.deps.is_empty()


@func
def merge_then_sigmoid(v: Tensor[(B, S, H, D), "f32"]) -> Tensor[(B, S, HD), "f32"]:
    w = reshape(v, new_shape=(B, S, HD))
    u = sigmoid(w)
    return u


def test_reshape_merge_pierces_via_div_mod():
    """The reverse direction (qwen3's ``attn_out`` merge back to
    ``[1,S,Q_PROJ]`` before ``w_o``): reconstructing the *old* split (h,d)
    from the *new* merged flat offset genuinely needs div/mod, unlike the
    split direction -- this is inherent to unpacking a flat index, not a
    design gap (see ``flat_reshape_map``'s docstring)."""
    tg = extract(merge_then_sigmoid)
    assert len(tg.units) == 1
    assert tg.units[0].name == "Sigmoid"

    print("\n=== reshape merge: reads ===")
    print(tg.reads)

    expected_reads = isl.map(
        f"{{ Sigmoid[b,s,e] -> v[b,s,floor(e/{D}),e mod {D}] : "
        f"0<=b<{B} and 0<=s<{S} and 0<=e<{HD} }}"
    )
    assert tg.reads.is_equal(expected_reads)
    assert "w[" not in str(tg.reads)


@func
def reshape_chain_then_sigmoid(
    x2: Tensor[(B, S, HD), "f32"],
) -> Tensor[(B, S, H, D, 1), "f32"]:
    y2 = reshape(x2, new_shape=(B, S, H, D))
    y3 = reshape(y2, new_shape=(B, S, H, D, 1))
    z2 = sigmoid(y3)
    return z2


def test_reshape_chain_pierces_through_both_hops():
    """A reshape of a reshape (split then insert a trailing unit axis):
    both hops fold away, and Sigmoid's read composes straight through to
    ``x2`` -- ``_buffer_namer``'s recursion chases the whole chain, not
    just one hop."""
    tg = extract(reshape_chain_then_sigmoid)
    assert len(tg.units) == 1
    op_kinds = [type(u.op.target).__name__ for u in tg.units]
    assert "Reshape" not in op_kinds

    print("\n=== reshape chain: reads ===")
    print(tg.reads)

    expected_reads = isl.map(
        f"{{ Sigmoid[b,s,h,d,u] -> x2[b,s,{D}*h+d] : "
        f"0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} and u=0 }}"
    )
    assert tg.reads.is_equal(expected_reads)
    assert "y2[" not in str(tg.reads) and "y3[" not in str(tg.reads)


@func
def reshape_multi_consumer(x3: Tensor[(B, S, HD), "f32"]) -> Tensor[(B, S, H, D), "f32"]:
    y4 = reshape(x3, new_shape=(B, S, H, D))
    z3 = sigmoid(y4)
    z4 = exp(y4)
    out = add(z3, z4)
    return out


def test_reshape_multiple_consumers_each_pierce_independently():
    """Two different ops (``Sigmoid``, ``Unary`` exp) reading the *same*
    reshape output each pierce independently to the same source buffer --
    the fold is a property of the reshape ``Call``, not of any one reader.
    """
    tg = extract(reshape_multi_consumer)
    op_names = {u.name for u in tg.units}
    assert op_names == {"Sigmoid", "Unary", "Binary"}

    print("\n=== reshape multi-consumer: reads ===")
    print(tg.reads)

    expected_sigmoid = isl.map(
        f"{{ Sigmoid[b,s,h,d] -> x3[b,s,{D}*h+d] : 0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} }}"
    )
    expected_unary = isl.map(
        f"{{ Unary[b,s,h,d] -> x3[b,s,{D}*h+d] : 0<=b<{B} and 0<=s<{S} and 0<=h<{H} and 0<=d<{D} }}"
    )
    assert expected_sigmoid.is_subset(tg.reads)
    assert expected_unary.is_subset(tg.reads)
    assert "y4[" not in str(tg.reads)


@func
def bare_reshape_return(x4: Tensor[(B, S, HD), "f32"]) -> Tensor[(B, S, H, D), "f32"]:
    y5 = reshape(x4, new_shape=(B, S, H, D))
    return y5


def test_boundary_reshape_with_no_consumer_fails_closed():
    """A body that is *nothing but* a reshape (``return reshape(x, ...)``,
    the boundary case the task calls out) has no compute op left once the
    reshape folds away -- `extract` fails closed with its existing
    empty-body error rather than fabricating a copy statement."""
    with pytest.raises(ExtractError, match="no compute ops to extract"):
        extract(bare_reshape_return)
