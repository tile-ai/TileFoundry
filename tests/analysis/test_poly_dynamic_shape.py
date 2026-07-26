"""``extract`` dynamic-shape (``DimVar``) coverage: a parametrised domain
used to make ``extract`` raise outright; now a ``DimVar`` axis flows straight
through as an isl parameter (``to_domain`` already produces one), and
resolves back to its ``ShapeDim`` in ``TileGraph.params``.

Checks: (1) the static gemm+rmsnorm HIR (``test_poly_model.py``) still
extracts unchanged; (2) a ``DimVar`` M-axis matmul extracts a parametrised
``TileGraph``; (3) that same dynamic graph schedules and emits a skeleton
with a symbolic loop bound.
"""
from __future__ import annotations

import re

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.render import emit_scaffold
from tilefoundry.schedule.select_atoms import select_atoms
from tilefoundry.target import CudaTarget


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_static_gemm_rmsnorm_extract_unchanged():
    """Static HIR still extracts the exact k-carry + MM->RN dependences
    ``test_poly_model.py`` validates; ``TileGraph.params`` stays empty."""
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


# Dynamic M axis: x:[seq,4] @ w:[4,2], seq a DimVar (real variable
# sequence length). N/K stay small so the expected extents below remain
# hand-checkable literals.
SEQ = DimVar("seq", 1, 128)


@func
def dyn_matmul(
    x: Tensor[(SEQ, 4), "bf16"],
    w: Tensor[(4, 2), "bf16"],
) -> Tensor[(SEQ, 2), "bf16"]:
    h = matmul(x, w)
    return h


def test_dynamic_matmul_extract_params_and_domain():
    """A DimVar M axis extracts a parametrised ``[seq]->{...}`` domain
    (``0 <= i < seq``, straight from ``to_domain`` -- no tiling), resolves
    ``TileGraph.params['seq']`` back to the exact ``DimVar``, and the M
    axis is still bounded (``dim_max_val`` a finite 126, not ``infty``,
    since ``seq``'s own half-open range ``[1, 128)`` tops out at 127).
    ``build_schedule_tree()`` stays parametrised too."""
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

    # M axis: 0 <= i < seq directly, so i's max possible value (maximizing
    # over every valid seq up to its own range's 127) is 126.
    assert int(mm_set.dim_min_val(0).num_si()) == 0
    assert int(mm_set.dim_max_val(0).num_si()) == 126
    # N=2, K=4: static axes, unaffected by M.
    assert int(mm_set.dim_max_val(1).num_si()) == 1
    assert int(mm_set.dim_max_val(2).num_si()) == 3

    tree = build_schedule_tree(tg)
    assert "[seq]" in str(tree)


def test_dynamic_matmul_end_to_end_emits_symbolic_loop():
    """extract -> build_schedule_tree -> emit_scaffold: the M loop's upper bound
    names the isl parameter directly, never a fixed integer trip count."""
    tg = extract(dyn_matmul)
    tree = build_schedule_tree(tg)
    skeleton, _swimlane, contracts = emit_scaffold(tree)

    print("\n=== dynamic matmul skeleton ===")
    print(skeleton.text)

    assert "HOLE_MM" in skeleton.text
    assert len(contracts) == 1

    m = re.search(r"for \(int c0 = 0; c0 <=? ([^;]+); c0 \+= 1\)", skeleton.text)
    assert m is not None, skeleton.text
    bound = m.group(1)
    assert "seq" in bound


def test_a_bounded_dimvar_goes_through_atom_selection_too():
    """A ``DimVar`` carries ``lo``/``hi``, so a parametrised extent still has a
    finite ``dim_max_val`` and every stage that needs an integer extent gets
    one -- atom selection included. This is the whole four-stage path, because
    the stage that needs static extents is the one the other dynamic tests
    stop short of."""
    assert (SEQ.lo, SEQ.hi) == (1, 128), "an unbounded DimVar is not constructible"

    solved = select_atoms(build_schedule_tree(extract(dyn_matmul)), target=CudaTarget())
    print("\n=== dynamic matmul decisions ===", solved.decisions["statements"])

    assert solved.decisions["status"] == "OPTIMAL"
    (statement,) = solved.decisions["statements"].values()
    # The extent the decisions carry is the parameter's own upper bound, not a
    # guessed trip count.
    assert statement["tile"][0] == SEQ.hi - 1
    emit_scaffold(solved)
