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
from tilefoundry.schedule.kernel_schedule import compute_schedule, outermost_band
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
    return compute_schedule(extract(fn))


def _bands(tree: "isl.schedule") -> list["isl.schedule_node_band"]:
    found: list["isl.schedule_node_band"] = []

    def visit(node) -> bool:
        if isinstance(node, isl.schedule_node_band):
            found.append(node)
        return True

    tree.get_root().foreach_descendant_top_down(visit)
    return found


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
    """The happy path, decision by decision: MM takes the SM80 atom, the
    tile is a whole multiple of that atom's shape on every band member and
    divides the band extent, the coincident extent is split over as many
    lanes as it has tiles, h -- the only buffer carrying a dependence --
    gets the ring, and the whole footprint fits the capacity fact."""
    solved = solve_resources(_scheduled(), target="cuda")
    assert isinstance(solved, TileGraph)
    decisions = solved.decisions
    print("\n=== decisions ===")
    for key, value in decisions.items():
        print(f"{key}: {value}")

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert decisions["makespan"] > 0

    extents = (64, 64, 128)
    tile, tiles = decisions["tile"], decisions["tiles"]
    for axis, (size, count) in enumerate(zip(tile, tiles)):
        assert size * count == extents[axis], (axis, size, count)
        assert size % _SM80_SHAPE[axis] == 0, (axis, size)

    mm = decisions["statements"]["MM"]
    assert mm["atom"] == _SM80_ATOM
    # A hole is one tile instance holding several atom calls, never one.
    assert mm["tile_atoms"] > 1
    assert mm["tile_atoms"] == (tile[0] // 16) * (tile[1] // 8) * (tile[2] // 16)

    rn = decisions["statements"]["RN"]
    assert rn["atom"] is None  # RMSNorm: no V1 atom candidate
    assert rn["start"] >= mm["end"]  # MM -> RN is a real RAW dependence on h

    # place is derived from the band's coincident members, not solved.
    assert decisions["coincident"] == (0,)
    assert mm["place"] == "coincident[0]" and rn["place"] == "coincident[0]"
    assert decisions["lane_split"] == builtins.min(decisions["lanes"], tiles[0])
    assert mm["lane_rounds"] == math.ceil(tiles[0] / decisions["lane_split"])

    # h carries MM's k accumulation and MM -> RN; nothing else carries one,
    # and no atom is async, so nothing else may buy a slot.
    assert solved.ring["h"] >= 2
    assert {buf: n for buf, n in solved.ring.items() if n > 1} == {"h": solved.ring["h"]}
    assert decisions["ring"] == solved.ring

    total = builtins.sum(decisions["footprint_bytes"].values())
    assert total <= decisions["capacity_bytes"], (total, decisions["capacity_bytes"])

    skeleton, _swimlane, _contracts = emit_scaffold(solved)
    print("\n=== skeleton ===")
    print(skeleton.text)
    assert f"% {solved.ring['h']}" in skeleton.text


def test_tiling_the_band_yields_a_tile_and_point_pair_at_atom_granularity():
    """AC-3-2 read off the returned tree, not off the decisions: the one
    band ``compute_schedule`` produced becomes two nested bands, and each
    tile band member's own stride is a whole multiple of the atom shape."""
    tg = _scheduled()
    assert len(_bands(tg.tree)) == 1

    solved = solve_resources(tg, target="cuda")
    bands = _bands(solved.tree)
    print("\n=== tiled schedule tree ===")
    print(solved.tree)

    assert len(bands) == 2, f"expected a tile band and a point band, got {len(bands)}"
    tile_band_node, point_band_node = bands
    assert point_band_node.get_ancestor_child_position(tile_band_node) == 0
    assert tile_band_node.n_member() == point_band_node.n_member() == 3

    extents = (64, 64, 128)
    for pos in range(3):
        n_tiles = _member_values(tile_band_node, solved.domain, pos)
        stride = extents[pos] // n_tiles
        assert stride == solved.decisions["tile"][pos], (pos, stride)
        assert stride % _SM80_SHAPE[pos] == 0, (pos, stride, _SM80_SHAPE[pos])
        assert _member_values(point_band_node, solved.domain, pos) == stride


def test_a_capacity_too_small_for_one_atom_tile_names_capacity_and_bytes():
    """AC-3-3: the smallest tile any solution can take is one atom on every
    band member; a capacity below that footprint is reported with both
    numbers, not as a bare infeasibility."""
    target = CudaTarget(device=_TunedDevice(shared_memory_per_cta_bytes=1024))
    with pytest.raises(SolveResourcesError) as error:
        solve_resources(_scheduled(), target=target)

    message = str(error.value)
    print("\n=== capacity error ===")
    print(message)
    assert "capacity 1024 bytes" in message
    # min tile (16, 8, 16): x 512 + w 256 + h 2048 + weight 256 + y 2048.
    assert "requests 5120 bytes" in message
    assert "h=2048" in message


def test_an_unsatisfiable_ring_lower_bound_reports_its_cause():
    """A capacity that holds one atom tile but not the ring depth the
    dependence distance forces comes back naming the smallest tile, its
    bytes and the carried distances -- V1 reports, it never reschedules."""
    target = CudaTarget(device=_TunedDevice(shared_memory_per_cta_bytes=6000))
    with pytest.raises(SolveResourcesError) as error:
        solve_resources(_scheduled(), target=target)

    message = str(error.value)
    print("\n=== infeasible cause ===")
    print(message)
    assert "INFEASIBLE" in message
    assert "5120 of 6000 capacity bytes" in message
    assert "'h': (0, 63, 1)" in message


def test_ring_depth_is_the_carried_tile_distance_plus_one():
    """isl reports h carried at distance 63 along band member 1 (MM writes
    ``h[i, j]`` at time ``(i, j, k)``, RN reads the whole row at ``(i, 63,
    127)``) and 1 along member 2 (the k accumulation). The ring must hold one
    more slot than the widest of those distances measured in tiles, and the
    objective must not buy a slot beyond it."""
    solved = solve_resources(_scheduled(), target="cuda")
    tile = solved.decisions["tile"]
    distance = builtins.max(math.ceil(63 / tile[1]), math.ceil(1 / tile[2]))
    print("\n=== tile ===", tile, "carried tiles", distance, "ring", solved.ring)
    assert solved.ring["h"] == distance + 1

    # The same bound is what makes a capacity of 6000 infeasible even though
    # one atom tile only needs 5120: see the previous test.


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
    print("\n=== load-bound ===", solved.decisions["tile"], solved.ring)
    print(solved.decisions["statements"]["MM"])

    assert builtins.max(solved.ring.values()) > 2
    mm = solved.decisions["statements"]["MM"]
    assert mm["load"] > mm["compute"], (mm["load"], mm["compute"])
    assert mm["stage"] < mm["compute"] + mm["load"]


def test_the_lane_split_never_claims_more_lanes_than_the_hardware_has():
    """The coincident extent is split across lanes, so a device with fewer
    lanes than coincident tiles has to run several rounds per lane."""
    solved = solve_resources(_scheduled(), target=CudaTarget(device=_TunedDevice(sm_count=2)))
    decisions = solved.decisions
    print("\n=== two lanes ===", decisions["tile"], decisions["tiles"], decisions["lane_split"])
    print(decisions["statements"]["MM"])

    assert decisions["lanes"] == 2
    assert decisions["lane_split"] == 2
    parallel = decisions["tiles"][0]
    for name in ("MM", "RN"):
        rounds = decisions["statements"][name]["lane_rounds"]
        assert rounds == math.ceil(parallel / 2), (name, rounds, parallel)


def test_the_model_footprint_matches_the_footprint_isl_counts():
    """The capacity constraint is fed a product of tile extents; here it is
    checked against isl's own count of the elements one tile instance
    touches, buffer by buffer."""
    solved = solve_resources(_scheduled(), target="cuda")
    tile = solved.decisions["tile"]
    band = outermost_band(compute_schedule(extract(bf16_gemm_rmsnorm)).tree)
    time_map = band.get_partial_schedule_union_map()

    origin = tuple(64 - tile[0] if i == 0 else 0 for i in range(3))
    dims = ", ".join(f"i{i}" for i in range(3))
    bounds = " and ".join(
        f"{origin[i]} <= i{i} <= {origin[i] + tile[i] - 1}" for i in range(3)
    )
    box = isl.set(f"{{ [{dims}] : {bounds} }}")

    elem_bytes = {"x": 2, "w": 2, "h": 2, "y": 2, "weight": 4}
    exact: dict[str, int] = {}
    for um in (solved.reads, solved.writes):
        maps: list["isl.map"] = []
        um.foreach_map(maps.append)
        for m in maps:
            buf = m.get_tuple_name(isl.dim_type.OUT)
            timed: list["isl.map"] = []
            m.apply_domain(time_map).foreach_map(timed.append)
            touched = timed[0].intersect_domain(box).range()
            previous = exact.get(buf)
            exact[buf] = touched if previous is None else previous.union(touched)

    counted = {
        buf: int(s.count_val().num_si()) * elem_bytes[buf] * solved.ring[buf]
        for buf, s in exact.items()
    }
    print("\n=== model  ===", solved.decisions["footprint_bytes"])
    print("=== isl    ===", counted)
    assert counted == solved.decisions["footprint_bytes"]


def test_no_candidate_statements_still_solve_with_a_default_duration():
    """Robustness: an all-f32 gemm+rmsnorm has zero atom candidates for
    either statement (MM: the sole SM80 atom is bf16-only; RN: no V1
    catalogue entry at all). No atom then constrains the tile granularity,
    but the tile, lane split and ring decisions still have to come out."""
    solved = solve_resources(_scheduled(f32_gemm_rmsnorm), target="cuda")
    decisions = solved.decisions
    print("\n=== f32 decisions ===", decisions["tile"], decisions["lane_split"], solved.ring)

    assert decisions["status"] in ("OPTIMAL", "FEASIBLE")
    assert decisions["statements"]["MM"]["atom"] is None
    assert decisions["statements"]["RN"]["atom"] is None
    assert decisions["statements"]["MM"]["tile_atoms"] is None
    assert decisions["makespan"] > 0
    for axis, extent in enumerate((64, 64, 128)):
        assert decisions["tile"][axis] * decisions["tiles"][axis] == extent
    assert solved.ring["h"] >= 2
    assert builtins.sum(decisions["footprint_bytes"].values()) <= decisions["capacity_bytes"]


def test_default_target_resolution_matches_explicit_cuda_target():
    """``target=None`` resolves via ``default_target()`` -- same convention
    as the CUDA target's own ``candidate_atoms``."""
    implicit = solve_resources(_scheduled())
    explicit = solve_resources(_scheduled(), target=default_target())

    assert implicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert explicit.decisions["statements"]["MM"]["atom"] == _SM80_ATOM
    assert implicit.decisions["tile"] == explicit.decisions["tile"]


def test_empty_tile_graph_raises_clear_error():
    """``tg.units == ()`` is an explicit, actionable error, never a silent
    no-op or a confusing CP-SAT/isl traceback."""
    tg = _scheduled()
    empty = replace(tg, units=())
    with pytest.raises(SolveResourcesError, match="units is empty"):
        solve_resources(empty, target="cuda")


def test_solving_before_compute_schedule_raises_clear_error():
    with pytest.raises(SolveResourcesError, match="compute_schedule"):
        solve_resources(extract(bf16_gemm_rmsnorm), target="cuda")


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
