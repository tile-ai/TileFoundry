"""``compute_schedule(TileGraph) -> TileGraph`` -- the isl affine schedule
computed over the polyhedral facts ``tests/analysis/test_poly_model.py``
pins, for the same minimal gemm+rmsnorm HIR ``Function``: it must cover the
whole domain, order every dependence strictly, and fuse the two statements.
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import compute_schedule


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def _lex_positive(rank: int) -> "isl.set":
    """`{ d : first non-zero component of d is positive }`, spelled out."""
    dims = ", ".join(f"d{i}" for i in range(rank))
    pieces = []
    for k in range(rank):
        zeros = [f"d{i} = 0" for i in range(k)]
        pieces.append(f"[{dims}] : " + " and ".join(zeros + [f"d{k} > 0"]))
    return isl.set("{ " + "; ".join(pieces) + " }")


def _lex_nonpositive(deltas: "isl.union_set") -> "isl.union_set":
    """The deltas that are not lexicographically positive."""
    out = isl.union_set("{}")
    pieces: list = []
    deltas.foreach_set(pieces.append)
    for piece in pieces:
        rank = piece.dim(isl.dim_type.SET)
        out = out.union(piece.subtract(_lex_positive(rank)))
    return out


def _outer_dimension(sched_map: "isl.union_map") -> "isl.union_map":
    """`sched_map` with every time dimension but the outermost projected out."""
    out = isl.union_map("{}")
    maps: list = []
    sched_map.foreach_map(maps.append)
    for m in maps:
        n_out = m.dim(isl.dim_type.OUT)
        out = out.union(m.project_out(isl.dim_type.OUT, 1, n_out - 1))
    return out


def test_schedule_is_legal_and_fuses_mm_with_rn():
    """``compute_schedule()`` covers exactly ``tg``'s domain, orders every dependence
    strictly, and puts MM and RN in one band.

    Both checks read the schedule as a relation, not as printed text. Legality
    maps each dependence into time and requires a lexicographically positive
    delta. Fusion requires the two statements to agree on the outermost time
    dimension: dropping ``set_validity`` still yields a legal schedule here,
    but splits the statements into a top-level sequence, which is what the
    fusion check catches.
    """
    tg = extract(gemm_rmsnorm)
    tree = compute_schedule(tg)

    print("\n=== gemm+rmsnorm isl schedule tree ===")
    print(tree)

    assert isinstance(tree, TileGraph)
    assert tree.tree.get_domain().is_equal(tg.domain)
    assert tree.ring == {}

    sched_map = tree.tree.get_map()
    timed = tg.deps.apply_domain(sched_map).apply_range(sched_map)
    assert not timed.is_empty(), "every dependence must survive into time space"
    print("=== dependence deltas in time space ===")
    print(timed.deltas())
    zero_or_negative = _lex_nonpositive(timed.deltas())
    assert zero_or_negative.is_empty(), (
        f"schedule violates a dependence: non-positive deltas {zero_or_negative}"
    )

    outer = _outer_dimension(sched_map)
    assert outer.is_equal(isl.union_map("{ MM[i,j,k] -> [i]; RN[i] -> [i] }")), (
        f"MM and RN must share the outermost time dimension, got {outer}"
    )
