"""``extract(HIR) -> TileGraph -> schedule(TileGraph) -> ScheduleTree`` for a
minimal gemm+rmsnorm HIR ``Function`` -- the agent-friendly compiler
path's polyhedral extraction + isl scheduling stage (``kernelize/``),
independent of the existing HIR -> TIR -> CUDA path (``compile.py`` /
``passes`` / ``codegen`` / ``ir`` are untouched by this module and this
test).

Extraction is plain element granularity (no tiling), so matmul's M/N/K
(2, 2, 4) *are* the iteration domain directly -- chosen at this small size
so the expected ``deps`` stay a direct, hand-checkable transcription of the
M1 de-risk probe (an isl ``union_access_info``/``compute_flow`` script over
gemm+rmsnorm reads/writes) and ``.tmp/poc/09_schedule_tree.py`` (mod isl's
``d0,d1,d2`` anonymous-tuple dim names vs. the reference's hand-picked
``i,j,k`` -- purely cosmetic).
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.kernelize import ScheduleTree, TileGraph, extract, schedule


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_extract_deps_include_k_carry_and_mm_to_rn():
    """One TileUnit per compute op (MM, RN), and ``deps`` auto-inferred
    from access relations alone (isl.union_access_info.compute_flow, per
    m1_deps_probe.py) reproduces exactly the two dependences that PoC
    hand-wrote: matmul's k-reduction carry and the MM -> RN RAW-on-h
    fusion edge -- nothing more, nothing less."""
    tg = extract(gemm_rmsnorm)
    assert isinstance(tg, TileGraph)

    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {"MM": "MatMul", "RN": "RMSNorm"}

    print("\n=== domain ===")
    print(tg.domain)
    print("=== reads ===")
    print(tg.reads)
    print("=== writes ===")
    print(tg.writes)
    print("=== deps (auto-inferred) ===")
    print(tg.deps)

    k_carry = isl.map("{ MM[i,j,k] -> MM[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<3 }")
    mm_to_rn = isl.map("{ MM[i,j,3] -> RN[i] : 0<=i<2 and 0<=j<2 }")
    assert k_carry.is_subset(tg.deps)
    assert mm_to_rn.is_subset(tg.deps)

    expected_total = isl.union_map("{}").union(k_carry).union(mm_to_rn)
    assert tg.deps.is_equal(expected_total)


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
    """``schedule()`` covers exactly ``tg``'s domain, orders every dependence
    strictly, and puts MM and RN in one band.

    Both checks read the schedule as a relation, not as printed text. Legality
    maps each dependence into time and requires a lexicographically positive
    delta. Fusion requires the two statements to agree on the outermost time
    dimension: dropping ``set_validity`` still yields a legal schedule here,
    but splits the statements into a top-level sequence, which is what the
    fusion check catches.
    """
    tg = extract(gemm_rmsnorm)
    tree = schedule(tg)

    print("\n=== gemm+rmsnorm isl schedule tree ===")
    print(tree)

    assert isinstance(tree, ScheduleTree)
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
