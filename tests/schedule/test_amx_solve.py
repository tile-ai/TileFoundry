"""``select_atoms(tg, target=<amx>)`` -- the resource decisions run against
the AMX target's own facts.

The subject is the fact plumbing, not a search: the picked atom has to
granularise the tile, and which atom is even a candidate has to follow from
the storage its operands need. The AMX outer product keeps its operands in the
X/Y/Z register files, so it is only a candidate for a statement small enough to
fit them; the NEON outer product streams through cache and is always one. An
untiled whole-tensor matmul therefore lands on NEON, which is the evidence that
the storage filter, not a cost model, is what separates the two.
"""
from __future__ import annotations

import builtins

import pytest

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule import Schedule, ScheduleOptions
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.select_atoms import AtomSelectionError, select_atoms
from tilefoundry.target import AmxTarget

_AMX_ATOM = "AMX_FMA32_16x16x1_F32"
_NEON_ATOM = "NEON_FMLA_4x4x1_F32"
_AMX_SHAPE = (16, 16, 1)
_NEON_SHAPE = (4, 4, 1)
_L1D_BYTES = 128 * 1024


@func(target="amx")
def f32_matmul(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def register_sized_f32_matmul(
    x: Tensor[(16, 8), "f32"],
    w: Tensor[(8, 16), "f32"],
) -> Tensor[(16, 16), "f32"]:
    h = matmul(x, w)  # noqa: F405
    return h


@func(target="amx")
def narrow_f32_matmul_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 16), "f32"],
    weight: Tensor[(16,), "f32"],
) -> Tensor[(64, 16), "f32"]:
    h = matmul(x, w)  # noqa: F405
    y = rms_norm(h, weight)  # noqa: F405
    return y


@func(target="amx")
def wide_f32_matmul_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 1024), "f32"],
    weight: Tensor[(1024,), "f32"],
) -> Tensor[(64, 1024), "f32"]:
    h = matmul(x, w)  # noqa: F405
    y = rms_norm(h, weight)  # noqa: F405
    return y


def _scheduled(fn=f32_matmul.entry_function()) -> TileGraph:
    return build_schedule_tree(extract(fn))


def _solve(fn=f32_matmul.entry_function()) -> TileGraph:
    return select_atoms(_scheduled(fn), target=AmxTarget(), stage="core")


def _mm(solved: TileGraph) -> dict:
    """The matmul statement's own decisions -- tile, atom and footprint are all
    per statement now that each one carries its own band."""
    return solved.decisions["statements"]["MM"]


def test_an_untiled_matmul_lands_on_the_neon_atom_not_the_amx_one():
    """The whole point of the storage filter: this matmul's operands are tens of
    kilobytes, so the AMX outer product -- which addresses its operands as X/Y/Z
    registers -- is not even a candidate, while NEON streaming through cache is.
    The tile is the statement's own extent and a whole multiple of NEON's."""
    solved = _solve()
    decisions = solved.decisions
    print("\n=== amx decisions ===")
    for key, value in decisions.items():
        print(f"{key}: {value}")

    assert decisions["status"] == "OPTIMAL"
    assert decisions["makespan"] > 0

    mm = decisions["statements"]["MM"]
    assert mm["atom"] == _NEON_ATOM
    assert mm["candidates"] == (_NEON_ATOM,)
    assert mm["tile"] == (64, 64, 128) and mm["tiles"] == (1, 1, 1)
    for axis, size in enumerate(mm["tile"]):
        assert size % _NEON_SHAPE[axis] == 0, (axis, size)
    # A hole is one tile instance holding several atom calls, never one.
    assert mm["tile_atoms"] == (64 // 4) * (64 // 4) * 128


def test_a_register_sized_matmul_keeps_both_atoms_so_the_pick_is_a_choice():
    """A 16x8 by 8x16 matmul holds its operands in 512 B of X, 512 B of Y and
    1 KiB of Z, so the AMX outer product clears the storage filter too and the
    statement has two candidates. Both are recorded; the pick is the first."""
    solved = _solve(register_sized_f32_matmul.entry_function())
    mm = _mm(solved)
    print("\n=== register-sized candidates ===", mm["candidates"], "picked", mm["atom"])

    assert mm["candidates"] == (_AMX_ATOM, _NEON_ATOM)
    assert mm["atom"] == _AMX_ATOM
    assert mm["tile"] == (16, 16, 8)
    assert mm["tile_atoms"] == (16 // 16) * (16 // 16) * 8


def test_the_capacity_recorded_against_the_footprint_is_the_core_l1d():
    """The store a core-level tile's working set lives in is that core's L1d.
    The AMX register files bound one atom instance instead, and do it by
    filtering that atom out of the candidates rather than by capping a tile."""
    solved = _solve()
    assert solved.decisions["capacity_bytes"] == _L1D_BYTES
    assert (
        solved.decisions["capacity_bytes"]
        == AmxTarget().device.l1d_bytes_per_performance_core
    )


def test_the_same_tile_graph_decides_a_different_atom_on_cuda():
    """One TileGraph, two targets: no registered CUDA atom takes f32 operands,
    so the CTA-level decisions name no atom at all where the AMX target's NEON
    entry granularises the same statement."""
    tg = _scheduled()
    on_amx = _mm(select_atoms(tg, target=AmxTarget(), stage="core"))
    on_cuda = _mm(select_atoms(tg, target="cuda", stage="cta"))
    print("\n=== amx  ===", on_amx["atom"], on_amx["tile"])
    print("=== cuda ===", on_cuda["atom"], on_cuda["tile"])

    assert on_amx["atom"] == _NEON_ATOM
    assert on_cuda["atom"] is None
    # The tile is the statement's own extent either way; only the atom differs.
    assert on_amx["tile"] == on_cuda["tile"]
    assert on_amx["tile_atoms"] is not None and on_cuda["tile_atoms"] is None


def test_an_op_outside_the_amx_catalogue_still_decides():
    """RMSNorm has no atom in this catalogue, so it granularises no tile member;
    the matmul in front of it still takes one and the dependence between them is
    still respected."""
    solved = _solve(narrow_f32_matmul_rmsnorm.entry_function())
    statements = solved.decisions["statements"]
    print("\n=== amx gemm+rmsnorm ===", solved.ring)
    print(statements)

    assert solved.decisions["status"] == "OPTIMAL"
    # 64 by 16 f32 accumulates in exactly the 4096-byte Z file, so this matmul
    # reaches the register-resident atom rather than the streaming one.
    assert statements["MM"]["atom"] == _AMX_ATOM
    assert statements["RN"]["atom"] is None and statements["RN"]["candidates"] == ()
    # Each statement is charged for the buffers it alone touches, at its own tile.
    assert set(statements["MM"]["footprint_bytes"]) == {"x", "w", "h"}
    assert set(statements["RN"]["footprint_bytes"]) == {"h", "weight", "y"}
    assert statements["RN"]["start"] >= statements["MM"]["end"]


def test_a_footprint_past_the_core_l1d_is_recorded_rather_than_refused():
    """A 1024-wide untiled statement holds hundreds of kilobytes at once -- well
    past the 128 KiB L1d, on both statements. That is recorded per buffer with
    the bytes each one asks for, not raised: a schedule over capacity is a bad
    schedule, not an absent one."""
    solved = _solve(wide_f32_matmul_rmsnorm.entry_function())
    statements = solved.decisions["statements"]
    print("\n=== wide decisions ===")
    for name, stmt in statements.items():
        print(name, stmt["footprint_bytes"], stmt["fits_capacity"])

    assert solved.decisions["status"] == "OPTIMAL"
    assert statements["MM"]["atom"] == _NEON_ATOM
    # h is the one buffer with a ring, so it is charged for both its slots.
    assert statements["MM"]["footprint_bytes"] == {"x": 32768, "w": 524288, "h": 524288}
    assert statements["RN"]["footprint_bytes"] == {"h": 524288, "weight": 4096, "y": 262144}
    for name, stmt in statements.items():
        assert builtins.sum(stmt["footprint_bytes"].values()) > _L1D_BYTES, name
        assert stmt["fits_capacity"] is False, name


def test_a_stage_the_target_does_not_serve_is_named():
    """The capacity and the catalogue are both projected for the level being
    decided, so asking for a level the AMX target does not schedule is reported
    as that rather than escaping as a bare projection failure."""
    with pytest.raises(AtomSelectionError, match="at stage 'cta'"):
        select_atoms(_scheduled(), target=AmxTarget(), stage="cta")


def test_the_bound_core_schedule_service_reports_the_nominal_makespan():
    """The service the target binds at its own topology level runs the same
    decisions and reports the nominal time in ns."""
    module = f32_matmul
    service = module.resolve_target().service(Schedule, "core")
    assert service is module.resolve_target().service(Schedule, "core")

    report = service.solve(
        module, module.entry_function(), ScheduleOptions(timeout_seconds=30),
    ).report
    print("\n=== report ===", report.to_json())

    assert report.stage == "core"
    assert report.target == "amx"
    assert report.root == "f32_matmul"
    assert report.status == "OPTIMAL"
    assert report.unit == "ns"
    assert report.selected > 0
    assert report.best_bound == report.selected
    assert report.gap == 0.0


def test_the_core_schedule_service_rejects_a_root_it_does_not_own():
    """The service decides for the target that owns it, and for the module's own
    entry function."""
    service = f32_matmul.resolve_target().service(Schedule, "core")
    other = narrow_f32_matmul_rmsnorm
    with pytest.raises(ValueError, match="module.entry_function"):
        service.solve(other, f32_matmul.entry_function())
    with pytest.raises(TypeError, match="HIR Module"):
        service.solve(f32_matmul.entry_function(), f32_matmul.entry_function())
