"""``solve_resources(tg, sched, target) -> ScheduleTree`` -- M3 (direction
C)'s core CP-SAT resource-decision block: over M1's isl schedule tree for
a bf16 gemm+rmsnorm HIR ``Function`` (bf16 so MM has a real SM80 atom
candidate, mirroring ``test_target_facts.py``'s own dtype note), choose
each statement's atom / lane placement / per-buffer ring depth, minimize
a coarse makespan, and hang the decoded decisions back onto the tree as
one top-level ``"DECISIONS"`` isl mark -- then feed the result into M2's
``emit_scaffold`` end to end and confirm the solved ring depths actually
show up as ``% N`` indexing in the rendered skeleton (proof the CP-SAT
ring decision really flows all the way to the scaffold, not just to
``ScheduleTree.ring`` in isolation).
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.kernelize import (
    ScheduleTree,
    SolveResourcesError,
    emit_scaffold,
    extract,
    schedule,
    solve_resources,
)
from tilefoundry.target import default_target


@func(target="cuda")
def bf16_gemm_rmsnorm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


@func
def f32_gemm_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_solve_resources_picks_atom_places_and_rings_end_to_end():
    """The full happy path: MM (bf16) gets the SM80 atom CP-SAT chose,
    RN (RMSNorm, no V1 atom candidate) still gets a place + falls back to
    a default duration, every written buffer (h, y) gets a ring depth, the
    decisions are hung on the tree as a mark, and emit_scaffold renders
    that ring depth as real `% N` indexing in the skeleton."""
    tg = extract(bf16_gemm_rmsnorm)
    tree = schedule(tg)

    solved = solve_resources(tg, tree, target="cuda")
    assert isinstance(solved, ScheduleTree)

    dumped = str(solved.tree)
    print("\n=== solved schedule tree (with DECISIONS mark) ===")
    print(dumped)
    assert "mark" in dumped
    assert "DECISIONS" in dumped

    mark_id = solved.tree.get_root().child(0).get_id()
    assert mark_id.name() == "DECISIONS"
    decisions = mark_id.user()
    print("\n=== decoded decisions (mark payload) ===")
    print(decisions)

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert decisions["makespan"] > 0

    mm = decisions["statements"]["MM"]
    assert mm["atom"] == "SM80_16x8x16_F32BF16BF16F32_TN"
    lanes = default_target().device.sm_count
    assert 0 <= mm["place"] < lanes
    assert mm["end"] > mm["start"]

    rn = decisions["statements"]["RN"]
    assert rn["atom"] is None  # RMSNorm: no V1 atom candidate, robust fallback
    assert 0 <= rn["place"] < lanes
    assert rn["end"] > rn["start"]
    # MM -> RN is a real dependence (RAW on h); the deps-chain makespan
    # model must serialize them, RN starting no earlier than MM ends.
    assert rn["start"] >= mm["end"]

    assert solved.ring, "ScheduleTree.ring must be non-empty: buffer -> depth"
    print("\n=== solved ring (buffer -> depth) ===")
    print(solved.ring)
    assert set(solved.ring) == {"h", "y"}
    for depth in solved.ring.values():
        assert 2 <= depth <= 4
    assert decisions["ring"] == solved.ring

    skeleton, swimlane, contracts = emit_scaffold(solved, tg)
    print("\n=== skeleton (ring-indexed, fed by solve_resources' output) ===")
    print(skeleton.text)
    assert any(f"% {depth}" in skeleton.text for depth in solved.ring.values())


def test_no_candidate_statements_fall_back_to_default_duration_without_crashing():
    """Robustness: an all-f32 gemm+rmsnorm has *zero* atom candidates for
    either statement (MM: the sole SM80 atom is bf16-only; RN: no V1 atom
    catalogue entry at all) -- solve_resources must still solve (default
    duration per statement, no atom `pick` vars at all), not raise."""
    tg = extract(f32_gemm_rmsnorm)
    tree = schedule(tg)

    solved = solve_resources(tg, tree, target="cuda")
    decisions = solved.tree.get_root().child(0).get_id().user()

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert decisions["statements"]["MM"]["atom"] is None
    assert decisions["statements"]["RN"]["atom"] is None
    assert decisions["makespan"] > 0
    assert solved.ring and set(solved.ring) == {"h", "y"}


def test_default_target_resolution_matches_explicit_cuda_target():
    """``target=None`` resolves via ``default_target()`` -- same
    convention as ``target_facts.candidate_atoms``."""
    tg = extract(bf16_gemm_rmsnorm)
    tree = schedule(tg)

    implicit = solve_resources(tg, tree)
    explicit = solve_resources(tg, tree, target=default_target())

    implicit_decisions = implicit.tree.get_root().child(0).get_id().user()
    explicit_decisions = explicit.tree.get_root().child(0).get_id().user()
    assert implicit_decisions["statements"]["MM"]["atom"] == "SM80_16x8x16_F32BF16BF16F32_TN"
    assert explicit_decisions["statements"]["MM"]["atom"] == "SM80_16x8x16_F32BF16BF16F32_TN"


def test_empty_tile_graph_raises_clear_error():
    """``tg.units == ()`` is an explicit, actionable error, never a
    silent no-op or a confusing CP-SAT/isl traceback."""
    tg = extract(bf16_gemm_rmsnorm)
    tree = schedule(tg)
    empty_tg = type(tg)(
        domain=tg.domain, deps=tg.deps, reads=tg.reads, writes=tg.writes,
        units=(), params=tg.params,
    )
    with pytest.raises(SolveResourcesError):
        solve_resources(empty_tg, tree, target="cuda")
