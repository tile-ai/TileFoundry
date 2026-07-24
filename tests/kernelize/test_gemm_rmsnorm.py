"""``extract(HIR) -> TileGraph -> schedule(TileGraph) -> ScheduleTree`` for a
minimal gemm+rmsnorm HIR ``Function`` -- the agent-friendly compiler
path's polyhedral extraction + isl scheduling stage (``kernelize/``),
independent of the existing HIR -> TIR -> CUDA path (``compile.py`` /
``passes`` / ``codegen`` / ``ir`` are untouched by this module and this
test).

Shapes are chosen so the fixed V1 tile size (``kernelize.DEFAULT_TILE_SIZE
== 32``) divides matmul's M/N/K (64, 64, 128) into exactly the small demo
domain ``Ti,Tj,Tk = 2,2,4`` that the M1 de-risk probe (an isl
``union_access_info``/``compute_flow`` script over gemm+rmsnorm
reads/writes) and ``.tmp/poc/09_schedule_tree.py`` already validated by
hand -- so this test's expected ``deps`` are a direct transcription of
that reference output (mod isl's ``d0,d1,d2`` anonymous-tuple dim names
vs. the reference's hand-picked ``i,j,k`` -- purely cosmetic).
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.kernelize import TileGraph, extract, schedule


@func
def gemm_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "f32"]:
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


def test_schedule_dumps_a_fused_two_phase_tree():
    """``schedule()`` (isl.schedule_constraints on domain/deps, per PoC 09)
    dumps a schedule tree sequencing MM before RN -- the auto-inferred
    MM -> RN dependence forces that order, exactly like the PoC's
    hand-written validity relation did."""
    tg = extract(gemm_rmsnorm)
    tree = schedule(tg)

    dumped = str(tree)
    print("\n=== gemm+rmsnorm isl schedule tree ===")
    print(dumped)

    assert "MM[" in dumped and "RN[" in dumped
    assert "sequence" in dumped
    # MM's filter must precede RN's within the printed sequence node (RN
    # depends on MM's last k-step, never the reverse). The *domain* field
    # prints its union pieces in isl's own (unrelated) internal order, so
    # the ordering check is scoped to the "sequence: [...]" tail only.
    sequence_tail = dumped[dumped.index("sequence"):]
    assert sequence_tail.index("MM[") < sequence_tail.index("RN[")
    # V1 is affine-structure only -- no CP-SAT ring/resource decisions yet.
    assert tree.ring == {}
