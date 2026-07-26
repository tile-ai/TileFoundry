"""``select_atoms(tg, target) -> TileGraph`` -- the resource decisions
over the isl schedule tree of a bf16 gemm+rmsnorm HIR ``Function`` (bf16 so
MM has a real SM80 atom candidate, mirroring ``test_atom_facts.py``'s own
dtype note).

What is checked here is that every decision answers to a measured fact: the
picked atom granularises the statement, the tile is the statement's own
extent (an operation is written at the size one hole computes), each ring
depth answers to the dependence distance isl reports, and the footprint the
capacity is compared against is the one isl counts. The capacity is recorded
against that footprint, never enforced -- a tile too wide for the store still
has a schedule, only a worse one.
"""
from __future__ import annotations

import builtins
import math
from dataclasses import dataclass, replace

import isl
import pytest

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import band_statement, build_schedule_tree, schedule_bands
from tilefoundry.schedule.render import emit_scaffold
from tilefoundry.schedule.select_atoms import AtomSelectionError, select_atoms
from tilefoundry.target import default_target
from tilefoundry.target.base import Device
from tilefoundry.target.cuda.target import CudaTarget

_SM80_ATOM = "SM80_16x8x16_F32BF16BF16F32_TN"
_SM80_SHAPE = (16, 8, 16)
_MM_EXTENTS = (64, 64, 128)


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


@dataclass(frozen=True)
class _TunedDevice(Device):
    """An H200 with its shared-memory capacity dialled to order, so a test can
    put the recorded footprint over the store on purpose."""

    name: str = "tuned"
    sm_count: int = 132
    hbm_bandwidth_bytes_per_second: int = 4_800_000_000_000
    shared_memory_per_cta_bytes: int = 227 * 1024

    def peak_for(self, dtype) -> int:
        return 989_500_000_000_000


def _scheduled(fn=bf16_gemm_rmsnorm) -> TileGraph:
    return build_schedule_tree(extract(fn))


def _bands_of(tree: "isl.schedule", stmt: str) -> list["isl.schedule_node_band"]:
    """``stmt``'s own bands, in top-down order -- one before ``select_atoms``
    has tiled, a tile band and a point band after."""
    return [band for band in schedule_bands(tree) if band_statement(band) == stmt]


def _member_values(band: "isl.schedule_node_band", domain: "isl.union_set", pos: int) -> int:
    """How many distinct values band member ``pos`` takes over ``domain``."""
    n_member = band.n_member()
    m = band.get_partial_schedule_union_map().intersect_domain(domain)
    sets: list["isl.set"] = []
    m.range().foreach_set(sets.append)
    (box,) = sets
    keep = box.project_out(isl.dim_type.SET, pos + 1, n_member - pos - 1).project_out(
        isl.dim_type.SET, 0, pos
    )
    return int(keep.count_val().num_si())


def test_solve_decides_the_atom_tile_and_rings_end_to_end():
    """The happy path, decision by decision: MM takes the SM80 atom, its tile
    is its own extent and a whole multiple of that atom's shape on every
    dimension, h -- the only buffer carrying a dependence -- gets the ring, and
    every statement's footprint is recorded against the capacity fact."""
    solved = select_atoms(_scheduled(), target="cuda")
    assert isinstance(solved, TileGraph)
    decisions = solved.decisions
    print("\n=== decisions ===")
    for key, value in decisions.items():
        print(f"{key}: {value}")

    assert decisions["status"] == "OPTIMAL"
    assert decisions["makespan"] > 0

    mm = decisions["statements"]["MM"]
    assert mm["tile"] == _MM_EXTENTS and mm["tiles"] == (1, 1, 1)
    for axis, size in enumerate(mm["tile"]):
        assert size % _SM80_SHAPE[axis] == 0, (axis, size)
    assert mm["atom"] == _SM80_ATOM
    assert mm["candidates"] == (_SM80_ATOM,)
    # A hole is one tile instance holding several atom calls, never one.
    assert mm["tile_atoms"] > 1
    assert mm["tile_atoms"] == math.prod(
        size // atom for size, atom in zip(mm["tile"], _SM80_SHAPE)
    )

    rn = decisions["statements"]["RN"]
    assert rn["atom"] is None and rn["candidates"] == ()  # RMSNorm: no V1 atom candidate
    assert rn["tile"] == (64,)
    assert rn["start"] >= mm["end"]  # MM -> RN is a real RAW dependence on h

    # place is derived from parallel_dims, not solved: MM accumulates over k.
    assert mm["coincident"] == (0, 1) and rn["coincident"] == (0,)
    assert mm["place"] == "coincident[0,1]" and rn["place"] == "coincident[0]"

    # h carries MM's k accumulation; nothing else carries one, so nothing else
    # may buy a slot.
    assert solved.ring["h"] == 2
    assert {buf: n for buf, n in solved.ring.items() if n > 1} == {"h": 2}
    assert decisions["ring"] == solved.ring

    for name, stmt in decisions["statements"].items():
        total = builtins.sum(stmt["footprint_bytes"].values())
        assert stmt["fits_capacity"] is (total <= decisions["capacity_bytes"]), name
        assert stmt["fits_capacity"], (name, total)

    skeleton, _swimlane, _contracts = emit_scaffold(solved)
    print("\n=== skeleton ===")
    print(skeleton.text)
    assert f"% {solved.ring['h']}" in skeleton.text


def test_tiling_the_bands_yields_a_tile_and_point_pair_at_atom_granularity():
    """AC-3-2 read off the returned tree, not off the decisions: each band
    ``build_schedule_tree`` produced becomes two nested bands. The tile is the
    whole extent, so the tile band takes one value per member and the point
    band walks the extent -- at a whole multiple of the atom shape."""
    tg = _scheduled()
    assert len(schedule_bands(tg.tree)) == len(tg.units) == 2

    solved = select_atoms(tg, target="cuda")
    print("\n=== tiled schedule tree ===")
    print(solved.tree)
    assert len(schedule_bands(solved.tree)) == 4

    for stmt, extents, atom in (("MM", _MM_EXTENTS, _SM80_SHAPE), ("RN", (64,), None)):
        bands = _bands_of(solved.tree, stmt)
        assert len(bands) == 2, f"{stmt}: expected a tile and a point band, got {len(bands)}"
        tile_band_node, point_band_node = bands
        assert point_band_node.get_ancestor_child_position(tile_band_node) == 0
        assert tile_band_node.n_member() == point_band_node.n_member() == len(extents)

        for pos in range(len(extents)):
            assert _member_values(tile_band_node, solved.domain, pos) == 1
            stride = _member_values(point_band_node, solved.domain, pos)
            assert stride == extents[pos] == solved.decisions["statements"][stmt]["tile"][pos]
            if atom is not None:
                assert stride % atom[pos] == 0, (stmt, pos, stride, atom[pos])


def test_a_capacity_too_small_for_the_footprint_is_recorded_not_raised():
    """The capacity is a fact to present, not a gate: a store far below what
    one tile holds still yields a full set of decisions, with the overflow
    recorded per statement. V1 reports a bad schedule, it does not refuse one."""
    target = CudaTarget(device=_TunedDevice(shared_memory_per_cta_bytes=512))
    solved = select_atoms(_scheduled(), target=target)
    decisions = solved.decisions
    print("\n=== over-capacity decisions ===")
    for name, stmt in decisions["statements"].items():
        print(name, stmt["footprint_bytes"], stmt["fits_capacity"])

    assert decisions["capacity_bytes"] == 512
    assert decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    for name, stmt in decisions["statements"].items():
        assert builtins.sum(stmt["footprint_bytes"].values()) > 512, name
        assert stmt["fits_capacity"] is False, name


def test_ring_depth_is_the_carried_tile_distance_plus_one():
    """h carries MM's k accumulation at distance 1 along MM's own last
    dimension, and nothing else: MM -> RN crosses statements, which the
    sequenced tree orders outright rather than through a ring. The ring holds
    one more slot than that distance measured in tiles."""
    solved = select_atoms(_scheduled(), target="cuda")
    tile = solved.decisions["statements"]["MM"]["tile"]
    distance = math.ceil(1 / tile[2])
    print("\n=== MM tile ===", tile, "carried tiles", distance, "ring", solved.ring)
    assert solved.ring["h"] == distance + 1 == 2


def test_the_recorded_footprint_matches_the_footprint_isl_counts():
    """The footprint recorded against the capacity is a product of tile
    extents; here it is checked against isl's own count of the elements one
    tile instance of each statement touches, buffer by buffer. Every band is
    its statement's own identity, so the box is written straight in the
    statement's coordinates."""
    solved = select_atoms(_scheduled(), target="cuda")
    elem_bytes = {"x": 2, "w": 2, "h": 2, "y": 2, "weight": 4}

    for stmt, extents in (("MM", _MM_EXTENTS), ("RN", (64,))):
        tile = solved.decisions["statements"][stmt]["tile"]
        rank = len(extents)
        # A corner box: the last tile along dimension 0, the first elsewhere.
        origin = tuple(extents[0] - tile[0] if d == 0 else 0 for d in range(rank))
        dims = ", ".join(f"i{d}" for d in range(rank))
        bounds = " and ".join(
            f"{origin[d]} <= i{d} <= {origin[d] + tile[d] - 1}" for d in range(rank)
        )
        box = isl.set(f"{{ [{dims}] : {bounds} }}").set_tuple_name(stmt)

        exact: dict[str, "isl.set"] = {}
        for um in (solved.reads, solved.writes):
            maps: list["isl.map"] = []
            um.foreach_map(maps.append)
            for m in maps:
                if m.get_tuple_name(isl.dim_type.IN) != stmt:
                    continue
                buf = m.get_tuple_name(isl.dim_type.OUT)
                touched = m.intersect_domain(box).range()
                previous = exact.get(buf)
                exact[buf] = touched if previous is None else previous.union(touched)

        counted = {
            buf: int(s.count_val().num_si()) * elem_bytes[buf] * solved.ring[buf]
            for buf, s in exact.items()
        }
        print(f"\n=== {stmt} recorded ===", solved.decisions["statements"][stmt]["footprint_bytes"])
        print(f"=== {stmt} isl      ===", counted)
        assert counted == solved.decisions["statements"][stmt]["footprint_bytes"]


def test_no_candidate_statements_still_decide_with_a_default_duration():
    """Robustness: an all-f32 gemm+rmsnorm has zero atom candidates for either
    statement (MM: the sole SM80 atom is bf16-only; RN: no V1 catalogue entry at
    all). No atom then granularises the tile, but the tile, ring and timeline
    decisions still have to come out."""
    solved = select_atoms(_scheduled(f32_gemm_rmsnorm), target="cuda")
    decisions = solved.decisions
    mm, rn = decisions["statements"]["MM"], decisions["statements"]["RN"]
    print("\n=== f32 decisions ===", mm["tile"], rn["tile"], solved.ring, decisions["makespan"])

    assert decisions["status"] == "OPTIMAL"
    assert mm["atom"] is None and rn["atom"] is None
    assert mm["candidates"] == () and rn["candidates"] == ()
    assert mm["tile_atoms"] is None
    assert decisions["makespan"] > 0
    assert mm["tile"] == _MM_EXTENTS
    assert solved.ring["h"] == 2
    for name, stmt in decisions["statements"].items():
        assert stmt["fits_capacity"], (name, stmt["footprint_bytes"])


def test_default_target_resolution_matches_explicit_cuda_target():
    """``target=None`` resolves via ``default_target()`` -- same convention
    as the CUDA target's own ``candidate_atoms``."""
    implicit = select_atoms(_scheduled())
    explicit = select_atoms(_scheduled(), target=default_target())

    assert implicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert explicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert (
        implicit.decisions["statements"]["MM"]["tile"]
        == explicit.decisions["statements"]["MM"]["tile"]
    )


def test_empty_tile_graph_raises_clear_error():
    """``tg.units == ()`` is an explicit, actionable error, never a silent
    no-op or a confusing isl traceback."""
    tg = _scheduled()
    empty = replace(tg, units=())
    with pytest.raises(AtomSelectionError, match="units is empty"):
        select_atoms(empty, target="cuda")


def test_solving_before_build_schedule_tree_raises_clear_error():
    with pytest.raises(AtomSelectionError, match="build_schedule_tree"):
        select_atoms(extract(bf16_gemm_rmsnorm), target="cuda")


def test_resolving_an_already_tiled_tree_raises_clear_error():
    """``select_atoms`` decides over the untiled tree; handed its own
    output it names the band count rather than silently deciding over tile
    coordinates."""
    solved = select_atoms(_scheduled(), target="cuda")
    with pytest.raises(AtomSelectionError, match="band"):
        select_atoms(solved, target="cuda")


def test_a_statement_order_that_runs_against_a_dependence_is_rejected():
    """The nominal timeline is a prefix sum over ``tg.units``, so that order has
    to agree with every dependence isl reports between two statements --
    reversing it puts RN's start before the MM it reads from."""
    tg = extract(bf16_gemm_rmsnorm)
    reversed_units = build_schedule_tree(replace(tg, units=tuple(reversed(tg.units))))
    with pytest.raises(AtomSelectionError, match="orders 'MM' before 'RN'"):
        select_atoms(reversed_units, target="cuda")
