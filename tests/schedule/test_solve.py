"""``solve_resources(tg, target) -> TileGraph`` -- the CP-SAT resource
decision over the isl schedule tree of a bf16 gemm+rmsnorm HIR ``Function``
(bf16 so MM has a real SM80 atom candidate, mirroring
``test_atom_facts.py``'s own dtype note).

What is checked here is that every decision is a *decided* one: the atom
pick drives the tile granularity, the tile size divides the band extent and
is a whole multiple of the picked atom's shape, the lane split is bounded by
the coincident tile count, and each ring depth answers to the dependence
distance isl reports, to the capacity fact and to the makespan. The last
test walks the CP-SAT model proto itself and fails on any variable no
constraint and not the objective reads.
"""
from __future__ import annotations

import builtins
import math
from dataclasses import dataclass, replace

import isl
import pytest
from google.protobuf import text_format
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import band_statement, build_schedule_tree, schedule_bands
from tilefoundry.schedule.render import emit_scaffold
from tilefoundry.schedule.solve_resources import SolveResourcesError, solve_resources
from tilefoundry.target import default_target
from tilefoundry.target.base import Device
from tilefoundry.target.cuda import service
from tilefoundry.target.cuda.target import CudaTarget

_SM80_ATOM = "SM80_16x8x16_F32BF16BF16F32_TN"
_SM80_SHAPE = (16, 8, 16)


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
    """An H200 with two facts dialled to order, so a test can put the model
    in the load-bound or the capacity-bound regime on purpose."""

    name: str = "tuned"
    sm_count: int = 132
    hbm_bandwidth_bytes_per_second: int = 4_800_000_000_000
    shared_memory_per_cta_bytes: int = 227 * 1024

    def peak_for(self, dtype) -> int:
        return 989_500_000_000_000


def _scheduled(fn=bf16_gemm_rmsnorm) -> TileGraph:
    return build_schedule_tree(extract(fn))


def _bands_of(tree: "isl.schedule", stmt: str) -> list["isl.schedule_node_band"]:
    """``stmt``'s own bands, in top-down order -- one before ``solve_resources``
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


@pytest.fixture
def solved_models(monkeypatch):
    """Every CP-SAT model ``solve_resources`` hands to a solver, in order."""
    seen: list[cp_model.CpModel] = []
    original = cp_model.CpSolver.Solve

    def spy(self, model, *args, **kwargs):
        seen.append(model)
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", spy)
    return seen


def test_solve_decides_atom_tile_lane_split_and_rings_end_to_end():
    """The happy path, decision by decision: MM takes the SM80 atom, its
    tile is a whole multiple of that atom's shape on every own dimension and
    divides its own extent, each statement's parallel extent is split over
    lanes, h -- the only buffer carrying a dependence -- gets the ring, and
    every statement's footprint fits the capacity fact."""
    solved = solve_resources(_scheduled(), target="cuda")
    assert isinstance(solved, TileGraph)
    decisions = solved.decisions
    print("\n=== decisions ===")
    for key, value in decisions.items():
        print(f"{key}: {value}")

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert decisions["makespan"] > 0

    mm = decisions["statements"]["MM"]
    for axis, (size, count) in enumerate(zip(mm["tile"], mm["tiles"])):
        assert size * count == (64, 64, 128)[axis], (axis, size, count)
        assert size % _SM80_SHAPE[axis] == 0, (axis, size)
    assert mm["atom"] == _SM80_ATOM
    # A hole is one tile instance holding several atom calls, never one.
    assert mm["tile_atoms"] > 1
    assert mm["tile_atoms"] == math.prod(
        size // atom for size, atom in zip(mm["tile"], _SM80_SHAPE)
    )

    rn = decisions["statements"]["RN"]
    assert rn["atom"] is None  # RMSNorm: no V1 atom candidate
    assert rn["tile"][0] * rn["tiles"][0] == 64
    assert rn["start"] >= mm["end"]  # MM -> RN is a real RAW dependence on h

    # place is derived from parallel_dims, not solved: MM accumulates over k.
    assert mm["coincident"] == (0, 1) and rn["coincident"] == (0,)
    assert mm["place"] == "coincident[0,1]" and rn["place"] == "coincident[0]"
    assert 1 <= decisions["lane_split"] <= decisions["lanes"]
    for stmt in (mm, rn):
        parallel = math.prod(stmt["tiles"][d] for d in stmt["coincident"])
        assert stmt["lane_rounds"] == math.ceil(parallel / decisions["lane_split"])

    # h carries MM's k accumulation; nothing else carries one, and no atom is
    # async, so nothing else may buy a slot.
    assert solved.ring["h"] >= 2
    assert {buf: n for buf, n in solved.ring.items() if n > 1} == {"h": solved.ring["h"]}
    assert decisions["ring"] == solved.ring

    for name, stmt in decisions["statements"].items():
        total = builtins.sum(stmt["footprint_bytes"].values())
        assert total <= decisions["capacity_bytes"], (name, total)

    skeleton, _swimlane, _contracts = emit_scaffold(solved)
    print("\n=== skeleton ===")
    print(skeleton.text)
    assert f"% {solved.ring['h']}" in skeleton.text


def test_tiling_the_bands_yields_a_tile_and_point_pair_at_atom_granularity():
    """AC-3-2 read off the returned tree, not off the decisions: each band
    ``build_schedule_tree`` produced becomes two nested bands, and each tile
    band member's own stride is a whole multiple of the atom shape."""
    tg = _scheduled()
    assert len(schedule_bands(tg.tree)) == len(tg.units) == 2

    solved = solve_resources(tg, target="cuda")
    print("\n=== tiled schedule tree ===")
    print(solved.tree)
    assert len(schedule_bands(solved.tree)) == 4

    for stmt, extents, atom in (("MM", (64, 64, 128), _SM80_SHAPE), ("RN", (64,), None)):
        bands = _bands_of(solved.tree, stmt)
        assert len(bands) == 2, f"{stmt}: expected a tile and a point band, got {len(bands)}"
        tile_band_node, point_band_node = bands
        assert point_band_node.get_ancestor_child_position(tile_band_node) == 0
        assert tile_band_node.n_member() == point_band_node.n_member() == len(extents)

        for pos in range(len(extents)):
            n_tiles = _member_values(tile_band_node, solved.domain, pos)
            stride = extents[pos] // n_tiles
            assert stride == solved.decisions["statements"][stmt]["tile"][pos], (stmt, pos)
            assert _member_values(point_band_node, solved.domain, pos) == stride
            if atom is not None:
                assert stride % atom[pos] == 0, (stmt, pos, stride, atom[pos])


def test_a_capacity_too_small_for_one_atom_tile_names_capacity_and_bytes():
    """AC-3-3: the smallest tile any solution can take is one atom on every
    dimension of the statement; a capacity below that footprint is reported
    with the statement and both numbers, not as a bare infeasibility."""
    target = CudaTarget(device=_TunedDevice(shared_memory_per_cta_bytes=512))
    with pytest.raises(SolveResourcesError) as error:
        solve_resources(_scheduled(), target=target)

    message = str(error.value)
    print("\n=== capacity error ===")
    print(message)
    assert "capacity 512 bytes" in message
    assert "of statement 'MM'" in message
    # MM's own min tile (16, 8, 16): x 512 + w 256 + h 256.
    assert "requests 1024 bytes" in message
    assert "x=512" in message


def test_an_unsatisfiable_ring_lower_bound_reports_its_cause():
    """A capacity that holds one atom tile but not the ring depth the
    dependence distance forces comes back naming each statement's smallest
    footprint and the carried distances -- V1 reports, it never reschedules."""
    target = CudaTarget(device=_TunedDevice(shared_memory_per_cta_bytes=1024))
    with pytest.raises(SolveResourcesError) as error:
        solve_resources(_scheduled(), target=target)

    message = str(error.value)
    print("\n=== infeasible cause ===")
    print(message)
    assert "INFEASIBLE" in message
    assert "capacity 1024 bytes" in message
    assert "'MM': 1024" in message
    assert "'MM': {'h': (0, 0, 1)}" in message


def test_ring_depth_is_the_carried_tile_distance_plus_one():
    """h carries MM's k accumulation at distance 1 along MM's own last
    dimension, and nothing else: MM -> RN crosses statements, which the
    sequenced tree orders outright rather than through a ring. The ring holds
    one more slot than that distance measured in tiles, and the objective
    must not buy a slot beyond it."""
    solved = solve_resources(_scheduled(), target="cuda")
    tile = solved.decisions["statements"]["MM"]["tile"]
    distance = math.ceil(1 / tile[2])
    print("\n=== MM tile ===", tile, "carried tiles", distance, "ring", solved.ring)
    assert solved.ring["h"] == distance + 1 == 2

    # The same bound is what makes a capacity of 1024 infeasible even though
    # MM's own smallest tile only needs 1024: see the previous test.


def test_is_async_decides_whether_a_read_buffer_may_ring_at_all(monkeypatch):
    """``AtomFact.is_async`` gates the ring: with the synchronous catalogue
    only ``h`` (which carries a dependence) may ring, so MM's loads stay
    exposed. Make the same atom async and the model buys slots for x and w
    and hides those loads, which is visible in the makespan."""
    sync = solve_resources(_scheduled(), target="cuda")
    assert sync.ring["x"] == 1 and sync.ring["w"] == 1
    mm_sync = sync.decisions["statements"]["MM"]
    assert mm_sync["stage"] == mm_sync["compute"] + mm_sync["load"]

    original = service.candidate_atoms
    monkeypatch.setattr(
        service,
        "candidate_atoms",
        lambda op, target: [replace(f, is_async=True) for f in original(op, target)],
    )
    asyn = solve_resources(_scheduled(), target="cuda")
    print("\n=== sync ring ===", sync.ring, sync.decisions["makespan"])
    print("=== async ring ===", asyn.ring, asyn.decisions["makespan"])

    assert asyn.ring["x"] > 1 and asyn.ring["w"] > 1
    mm_async = asyn.decisions["statements"]["MM"]
    assert mm_async["stage"] < mm_async["compute"] + mm_async["load"]
    assert asyn.decisions["makespan"] < sync.decisions["makespan"]


def test_a_load_bound_device_buys_more_than_two_ring_slots(monkeypatch):
    """Two slots hide one tile of load behind one tile of compute. When the
    load is several tiles of compute long, the only way to hide it is more
    slots, so the model must reach past a plain double buffer."""
    original = service.candidate_atoms
    monkeypatch.setattr(
        service,
        "candidate_atoms",
        lambda op, target: [replace(f, is_async=True) for f in original(op, target)],
    )
    slow = CudaTarget(device=_TunedDevice(hbm_bandwidth_bytes_per_second=100_000_000_000))
    solved = solve_resources(_scheduled(), target=slow)
    print("\n=== load-bound ===", solved.ring)
    print(solved.decisions["statements"]["MM"])

    assert builtins.max(solved.ring.values()) > 2
    mm = solved.decisions["statements"]["MM"]
    assert mm["load"] > mm["compute"], (mm["load"], mm["compute"])
    assert mm["stage"] < mm["compute"] + mm["load"]


def test_the_lane_split_never_claims_more_lanes_than_the_hardware_has():
    """Each statement's parallel extent is split across lanes, so a device
    with fewer lanes than parallel tiles has to run several rounds per lane."""
    solved = solve_resources(_scheduled(), target=CudaTarget(device=_TunedDevice(sm_count=2)))
    decisions = solved.decisions
    print("\n=== two lanes ===", decisions["lane_split"])
    print(decisions["statements"]["MM"])

    assert decisions["lanes"] == 2
    assert decisions["lane_split"] == 2
    for name in ("MM", "RN"):
        stmt = decisions["statements"][name]
        parallel = math.prod(stmt["tiles"][d] for d in stmt["coincident"])
        assert stmt["lane_rounds"] == math.ceil(parallel / 2), (name, parallel)


def test_the_model_footprint_matches_the_footprint_isl_counts():
    """The capacity constraint is fed a product of tile extents; here it is
    checked against isl's own count of the elements one tile instance of each
    statement touches, buffer by buffer. Every band is its statement's own
    identity, so the box is written straight in the statement's coordinates."""
    solved = solve_resources(_scheduled(), target="cuda")
    elem_bytes = {"x": 2, "w": 2, "h": 2, "y": 2, "weight": 4}

    for stmt, extents in (("MM", (64, 64, 128)), ("RN", (64,))):
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
        print(f"\n=== {stmt} model ===", solved.decisions["statements"][stmt]["footprint_bytes"])
        print(f"=== {stmt} isl   ===", counted)
        assert counted == solved.decisions["statements"][stmt]["footprint_bytes"]


def test_no_candidate_statements_still_solve_with_a_default_duration():
    """Robustness: an all-f32 gemm+rmsnorm has zero atom candidates for
    either statement (MM: the sole SM80 atom is bf16-only; RN: no V1
    catalogue entry at all). No atom then constrains the tile granularity,
    but the tile, lane split and ring decisions still have to come out."""
    solved = solve_resources(_scheduled(f32_gemm_rmsnorm), target="cuda")
    decisions = solved.decisions
    mm, rn = decisions["statements"]["MM"], decisions["statements"]["RN"]
    print("\n=== f32 decisions ===", mm["tile"], rn["tile"], decisions["lane_split"], solved.ring)

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert mm["atom"] is None and rn["atom"] is None
    assert mm["tile_atoms"] is None
    assert decisions["makespan"] > 0
    for axis, extent in enumerate((64, 64, 128)):
        assert mm["tile"][axis] * mm["tiles"][axis] == extent
    assert solved.ring["h"] >= 2
    for name, stmt in decisions["statements"].items():
        total = builtins.sum(stmt["footprint_bytes"].values())
        assert total <= decisions["capacity_bytes"], (name, total)


def test_default_target_resolution_matches_explicit_cuda_target():
    """``target=None`` resolves via ``default_target()`` -- same convention
    as the CUDA target's own ``candidate_atoms``."""
    implicit = solve_resources(_scheduled())
    explicit = solve_resources(_scheduled(), target=default_target())

    assert implicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert explicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert (
        implicit.decisions["statements"]["MM"]["tile"]
        == explicit.decisions["statements"]["MM"]["tile"]
    )


def test_empty_tile_graph_raises_clear_error():
    """``tg.units == ()`` is an explicit, actionable error, never a silent
    no-op or a confusing CP-SAT/isl traceback."""
    tg = _scheduled()
    empty = replace(tg, units=())
    with pytest.raises(SolveResourcesError, match="units is empty"):
        solve_resources(empty, target="cuda")


def test_solving_before_build_schedule_tree_raises_clear_error():
    with pytest.raises(SolveResourcesError, match="build_schedule_tree"):
        solve_resources(extract(bf16_gemm_rmsnorm), target="cuda")


def test_resolving_an_already_tiled_tree_raises_clear_error():
    """``solve_resources`` decides over the untiled tree; handed its own
    output it names the band count rather than silently deciding over tile
    coordinates."""
    solved = solve_resources(_scheduled(), target="cuda")
    with pytest.raises(SolveResourcesError, match="band"):
        solve_resources(solved, target="cuda")


# ---------------------------------------------------------------------------
# AC-3-1
# ---------------------------------------------------------------------------

# The CP-SAT proto fields that name a variable: `vars` inside a linear
# expression or constraint, `literals` inside a boolean one, and a
# constraint's own enforcement literals. A negative entry is the negation of
# variable `~entry`.
_VARIABLE_FIELDS = frozenset({"vars", "literals", "enforcement_literal"})


def _referenced(message) -> set[int]:
    found: set[int] = set()
    for field, value in message.ListFields():
        if field.name in _VARIABLE_FIELDS:
            found.update(v if v >= 0 else ~v for v in value)
        elif field.type == field.TYPE_MESSAGE:
            parts = value if field.label == field.LABEL_REPEATED else [value]
            for part in parts:
                found |= _referenced(part)
    return found


def _readable_proto(model: cp_model.CpModel):
    """``model``'s proto as the reflective Python message, so the walk above
    can be generic: the solver hands out a C++ binding with no reflection."""
    proto = cp_model_pb2.CpModelProto()
    text_format.Parse(str(model.Proto()), proto)
    return proto


def test_every_solved_variable_is_read_by_a_constraint_or_the_objective(solved_models):
    """AC-3-1, machine-checked against the model CP-SAT was handed: a
    variable that no constraint and not the objective mentions is a free
    variable that decodes to its bound, which is what this milestone exists
    to remove."""
    solve_resources(_scheduled(), target="cuda")
    (model,) = solved_models
    proto = _readable_proto(model)

    read: set[int] = set()
    for constraint in proto.constraints:
        read |= _referenced(constraint)
    read |= _referenced(proto.objective)

    unread = [
        (index, variable.name or f"<anonymous {index}>")
        for index, variable in enumerate(proto.variables)
        if index not in read
    ]
    print(f"\n=== {len(proto.variables)} variables, {len(proto.constraints)} constraints ===")
    print("read by objective:", sorted(_referenced(proto.objective)))
    assert not unread, f"variables only decoded, never constrained: {unread}"


def test_the_objective_reads_the_makespan_and_every_ring(solved_models):
    """The ring depths are not merely constrained, they are priced: the
    objective is the makespan first and the ring slack second, so a slot
    nothing overlaps is never bought."""
    solve_resources(_scheduled(), target="cuda")
    (model,) = solved_models
    proto = _readable_proto(model)

    names = {index: proto.variables[index].name for index in _referenced(proto.objective)}
    print("\n=== objective terms ===", sorted(names.values()))
    assert "makespan" in names.values()
    assert "lane_split" in names.values()
    assert {name for name in names.values() if name.startswith("ring_")} == {
        "ring_h", "ring_w", "ring_weight", "ring_x", "ring_y",
    }
