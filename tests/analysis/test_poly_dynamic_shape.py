"""``extract`` dynamic-shape (``DimVar``) coverage: a parametrised domain
used to make ``extract`` raise outright; now a ``DimVar`` axis flows straight
through as an isl parameter (``to_domain`` already produces one), and
resolves back to its ``ShapeDim`` in ``TileGraph.params``.

Two things have to hold for a decode kernel, and they are the two below: the
parameter stays *bounded* (a domain nothing bounds cannot be scheduled), and the
loop that comes out of it names the parameter instead of a trip count somebody
guessed. The static counterpart of this graph is pinned in
``test_analysis_invariants.py``.
"""
from __future__ import annotations

import re

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.render import emit_scaffold

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
    since ``seq``'s own half-open range ``[1, 128)`` tops out at 127 --
    an unbounded ``DimVar`` is not constructible in the first place).
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
    skeleton, _swimlane, contracts = emit_scaffold(tg, tree, {})

    print("\n=== dynamic matmul skeleton ===")
    print(skeleton.text)

    assert "HOLE_MM" in skeleton.text
    assert len(contracts) == 1

    m = re.search(r"for \(int c0 = 0; c0 <=? ([^;]+); c0 \+= 1\)", skeleton.text)
    assert m is not None, skeleton.text
    bound = m.group(1)
    assert "seq" in bound
