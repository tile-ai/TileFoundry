"""``extract`` dynamic-shape (``DimVar``) coverage: a parametrised domain
used to make ``extract`` raise outright; now it tiles a ``DimVar`` axis
via an isl-parameter bound instead of a fixed count, and resolves every
such parameter back to its ``ShapeDim`` in ``TileGraph.params``.

Checks: (1) the static gemm+rmsnorm HIR (``test_gemm_rmsnorm.py``) still
extracts unchanged; (2) a ``DimVar`` M-axis matmul extracts a parametrised
``TileGraph``; (3) that same dynamic graph schedules and emits a skeleton
with a symbolic loop bound; (4) optionally, ``Qwen3_1_7B.mlp``'s
non-dividing ``S_CAP=4`` now tiles at the real ``DEFAULT_TILE_SIZE``
without the old ``tile_size=1`` workaround.
"""
from __future__ import annotations

import re
from collections import Counter

import isl

from tests.models.qwen3_1_7b.qwen3_1_7b_module import Qwen3_1_7B
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.kernelize import TileGraph, emit_scaffold, extract, schedule


@func
def gemm_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_static_gemm_rmsnorm_extract_unchanged():
    """Static HIR still extracts the exact k-carry + MM->RN dependences
    ``test_gemm_rmsnorm.py`` validates; ``TileGraph.params`` stays empty."""
    tg = extract(gemm_rmsnorm)
    assert isinstance(tg, TileGraph)
    assert tg.params == {}
    assert tg.domain.space().dim(isl.dim_type.PARAM) == 0

    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {"MM": "MatMul", "RN": "RMSNorm"}

    k_carry = isl.map("{ MM[i,j,k] -> MM[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<3 }")
    mm_to_rn = isl.map("{ MM[i,j,3] -> RN[i] : 0<=i<2 and 0<=j<2 }")
    assert k_carry.is_subset(tg.deps)
    assert mm_to_rn.is_subset(tg.deps)
    expected_total = isl.union_map("{}").union(k_carry).union(mm_to_rn)
    assert tg.deps.is_equal(expected_total)


# Dynamic M axis: x:[seq,128] @ w:[128,64], seq a DimVar (real variable
# sequence length), N/K static.
SEQ = DimVar("seq", 1, 128)


@func
def dyn_matmul(
    x: Tensor[(SEQ, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(SEQ, 64), "bf16"]:
    h = matmul(x, w)
    return h


def test_dynamic_matmul_extract_params_and_domain():
    """A DimVar M axis extracts a parametrised ``[seq]->{...}`` domain,
    resolves ``TileGraph.params['seq']`` back to the exact ``DimVar``, and
    the M-tile axis is still bounded (``dim_max_val`` a finite 3, not
    ``infty``). ``schedule()`` stays parametrised too."""
    tg = extract(dyn_matmul)
    assert isinstance(tg, TileGraph)

    assert tg.params == {"seq": SEQ}
    assert tg.domain.space().dim(isl.dim_type.PARAM) == 1
    assert "[seq]" in str(tg.domain)
    assert "x[" in str(tg.reads) and "w[" in str(tg.reads)
    assert "h[" in str(tg.writes)

    sets: list = []
    tg.domain.foreach_set(sets.append)
    assert len(sets) == 1
    (mm_set,) = sets
    assert mm_set.get_tuple_name() == "MM"

    # M-tile axis: 0 <= ti and 32*ti < seq <= 127 => ti in [0, 3].
    assert int(mm_set.dim_min_val(0).num_si()) == 0
    assert int(mm_set.dim_max_val(0).num_si()) == 3
    # N=64 -> 2 tiles, K=128 -> 4 tiles: static axes, unaffected by M.
    assert int(mm_set.dim_max_val(1).num_si()) == 1
    assert int(mm_set.dim_max_val(2).num_si()) == 3

    tree = schedule(tg)
    assert "[seq]" in str(tree)


def test_dynamic_matmul_end_to_end_emits_symbolic_loop():
    """extract -> schedule -> emit_scaffold: the M-tile loop's upper bound
    names the isl parameter directly (isl's own ceildiv rendering), never
    a fixed integer trip count."""
    tg = extract(dyn_matmul)
    tree = schedule(tg)
    skeleton, _swimlane, contracts = emit_scaffold(tree, tg)

    print("\n=== dynamic matmul skeleton ===")
    print(skeleton.text)

    assert "HOLE_MM" in skeleton.text
    assert len(contracts) == 1

    m = re.search(r"for \(int c0 = 0; c0 <= ([^;]+); c0 \+= 1\)", skeleton.text)
    assert m is not None, skeleton.text
    bound = m.group(1)
    assert "seq" in bound
    assert "32" in bound


def test_mlp_ceildiv_tiles_non_dividing_s_cap_at_default_tile_size():
    """Qwen3_1_7B.mlp's real S_CAP=4 (not divisible by DEFAULT_TILE_SIZE)
    used to need tile_size=1 (test_extract_elementwise.py); ceiling
    division now tiles it at the real tile_size=32 directly."""
    tg = extract(Qwen3_1_7B.mlp)  # default tile_size == DEFAULT_TILE_SIZE (32)
    assert isinstance(tg, TileGraph)

    op_names = [type(u.op.target).__name__ for u in tg.units]
    assert Counter(op_names) == Counter(
        {"RMSNorm": 1, "MatMul": 3, "Sigmoid": 1, "Binary": 2}
    )
    assert len(tg.units) == 7
    assert tg.params == {}  # mlp's shapes are all static ints, no DimVar

    assert not tg.domain.is_empty()
    assert not tg.reads.is_empty()
    assert not tg.writes.is_empty()
    assert not tg.deps.is_empty()
