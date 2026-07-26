"""``solve_resources(tg, target=<amx>)`` -- the CP-SAT resource decision run
against the AMX target's own facts.

The subject is the fact plumbing, not the solver: the AMX candidate atom has to
granularise the tile, and the capacity the footprint is charged against has to
be the Z accumulator file (4096 B) rather than a GPU's shared memory. The last
two tests hold everything but that one capacity fact fixed, so a tile that grows
when it is widened is evidence that Z is what binds.
"""
from __future__ import annotations

import builtins

import pytest

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.ir.core.module import Module
from tilefoundry.schedule import Schedule, ScheduleOptions
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.solve_resources import SolveResourcesError, solve_resources
from tilefoundry.target import AmxTarget
from tilefoundry.target.amx import AppleM2Pro

_AMX_ATOM = "AMX_FMA32_16x16x1_F32"
_AMX_SHAPE = (16, 16, 1)
_Z_BYTES = 4096


@func(target="amx")
def f32_matmul(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
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


class _L1SizedTileStore(AppleM2Pro):
    """The same device with its tile store widened to the performance core's
    L1d, so a comparison against it changes the capacity fact and nothing else."""

    @property
    def l1_capacity_bytes(self) -> int:
        return self.l1d_bytes_per_performance_core


def _scheduled(fn=f32_matmul) -> TileGraph:
    return build_schedule_tree(extract(fn))


def _solve(fn=f32_matmul, device=None) -> TileGraph:
    target = AmxTarget() if device is None else AmxTarget(device=device)
    return solve_resources(_scheduled(fn), target=target, stage="core")


def _mm(solved: TileGraph) -> dict:
    """The matmul statement's own decisions -- tile, atom and footprint are all
    per statement now that each one carries its own band."""
    return solved.decisions["statements"]["MM"]


def test_the_amx_target_solves_to_optimal_at_atom_granularity():
    """The AMX f32 outer product is picked, MM's tile is a whole multiple of its
    16x16x1 shape on every own dimension and divides its own extent, and the
    footprint fits the capacity fact."""
    solved = _solve()
    decisions = solved.decisions
    print("\n=== amx decisions ===")
    for key, value in decisions.items():
        print(f"{key}: {value}")

    assert decisions["status"] == "OPTIMAL"
    assert decisions["makespan"] > 0

    mm = decisions["statements"]["MM"]
    extents = (64, 64, 128)
    for axis, (size, count) in enumerate(zip(mm["tile"], mm["tiles"])):
        assert size * count == extents[axis], (axis, size, count)
        assert size % _AMX_SHAPE[axis] == 0, (axis, size)

    assert mm["atom"] == _AMX_ATOM
    # A hole is one tile instance holding several atom calls, never one.
    assert mm["tile_atoms"] > 1
    assert builtins.sum(mm["footprint_bytes"].values()) <= decisions["capacity_bytes"]

    # Two AMX units, so the coincident tiles are spread over at most two lanes.
    assert decisions["lanes"] == 2
    assert decisions["lane_split"] <= 2


def test_the_capacity_the_footprint_is_charged_against_is_the_z_accumulator():
    """The tile-memory capacity fact is the 4096-byte Z file, not the 128 KiB
    L1d and not a GPU's shared memory."""
    solved = _solve()
    assert solved.decisions["capacity_bytes"] == _Z_BYTES
    assert solved.decisions["capacity_bytes"] == AmxTarget().device.amx_accumulator_bytes


def test_widening_only_the_capacity_fact_grows_the_tile():
    """Everything but the capacity held fixed: with Z the tile fits 4096 bytes,
    with the performance core's L1d in its place the same solve takes a strictly
    larger tile. That is what makes Z the fact that binds here."""
    on_z = _mm(_solve())
    on_l1d = _mm(_solve(device=_L1SizedTileStore()))
    z_bytes = builtins.sum(on_z["footprint_bytes"].values())
    l1d_bytes = builtins.sum(on_l1d["footprint_bytes"].values())
    print("\n=== Z    ===", on_z["tile"], z_bytes, _Z_BYTES)
    print("=== L1d  ===", on_l1d["tile"], l1d_bytes, 128 * 1024)

    assert z_bytes <= _Z_BYTES < l1d_bytes
    assert on_z["tile"] != on_l1d["tile"]
    assert any(z < l1d for z, l1d in zip(on_z["tile"], on_l1d["tile"]))


def test_the_same_tile_graph_solves_to_a_different_tile_on_cuda():
    """One TileGraph, two targets: the CUDA capacity fact is 227 KiB of shared
    memory and no registered CUDA atom takes f32 operands, so the tile it
    decides is unconstrained by an atom shape and far larger."""
    tg = _scheduled()
    on_amx = _mm(solve_resources(tg, target=AmxTarget(), stage="core"))
    on_cuda = _mm(solve_resources(tg, target="cuda", stage="cta"))
    print("\n=== amx  ===", on_amx["tile"], _Z_BYTES)
    print("=== cuda ===", on_cuda["tile"], 227 * 1024)

    assert on_cuda["atom"] is None
    assert on_amx["tile"] != on_cuda["tile"]


def test_an_op_outside_the_amx_catalogue_still_solves():
    """RMSNorm has no AMX atom, so it constrains no tile member; the matmul in
    front of it still takes the AMX atom and the dependence between them is
    still respected."""
    solved = _solve(narrow_f32_matmul_rmsnorm)
    statements = solved.decisions["statements"]
    print("\n=== amx gemm+rmsnorm ===", solved.ring)
    print(statements)

    assert solved.decisions["status"] == "OPTIMAL"
    assert statements["MM"]["atom"] == _AMX_ATOM
    assert statements["RN"]["atom"] is None
    # Each statement is charged for the buffers it alone touches, at its own tile.
    assert set(statements["MM"]["footprint_bytes"]) == {"x", "w", "h"}
    assert set(statements["RN"]["footprint_bytes"]) == {"h", "weight", "y"}
    assert statements["RN"]["start"] >= statements["MM"]["end"]
    for stmt in statements.values():
        assert builtins.sum(stmt["footprint_bytes"].values()) <= _Z_BYTES


def test_a_row_normalising_epilogue_too_wide_for_z_is_reported_with_both_numbers():
    """RMSNorm's reduction axis never enters its domain, so no tile of it can
    subdivide the row: one 1024-wide f32 row is 4096 bytes of its input, of its
    output and of its gamma alike -- three times the whole accumulator file at
    RMSNorm's own smallest tile. Reported naming the statement, the capacity and
    the bytes rather than as a bare infeasibility."""
    with pytest.raises(SolveResourcesError) as error:
        _solve(wide_f32_matmul_rmsnorm)

    message = str(error.value)
    print("\n=== amx capacity error ===")
    print(message)
    assert "capacity 4096 bytes cannot hold one atom tile (1,) of statement 'RN'" in message
    assert "requests 12288 bytes" in message
    assert "h=4096" in message and "y=4096" in message and "weight=4096" in message


def test_the_bound_core_schedule_service_reports_the_solved_makespan():
    """The service the target binds at its own topology level runs the same
    solve and reports the objective in ns."""
    module = Module("amx_matmul", (f32_matmul,), "f32_matmul")
    service = f32_matmul.target.service(Schedule, "core")
    assert service is f32_matmul.target.service(Schedule, "core")

    report = service.solve(module, f32_matmul, ScheduleOptions(timeout_seconds=30)).report
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
    """The service solves for the target that owns it, and for the module's own
    entry function."""
    service = f32_matmul.target.service(Schedule, "core")
    other = Module("other", (narrow_f32_matmul_rmsnorm,), "narrow_f32_matmul_rmsnorm")
    with pytest.raises(ValueError, match="module.entry_function"):
        service.solve(other, f32_matmul)
    with pytest.raises(TypeError, match="HIR Module"):
        service.solve(f32_matmul, f32_matmul)
