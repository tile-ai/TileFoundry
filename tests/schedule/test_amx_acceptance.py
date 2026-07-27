"""The whole path over real decoder-layer kernels on the AMX target.

Every other test in this package pins one mechanism on a fixture built to
exercise it. This one runs the four stages the CLI runs -- ``extract`` ->
``build_schedule_tree`` -> ``select_atoms`` -> ``emit_scaffold`` -- over the
``Qwen3_1_7B`` kernels as authored, and checks the four properties a usable
result has to have: the schedule respects every dependence, the picked atom
granularises the tile, every statement is either granularised or recorded as
uncovered, and each hole's working set is reported against the level's store.

The pair that matters is ``mlp`` against ``tiled_mlp``. They compute the same
value; only the second is written as the loop nest whose blocks fit the AMX
register files. The first therefore reaches only the cache-streaming NEON
atom and the second reaches the register-resident AMX one -- which is the
end-to-end evidence that operand storage, not a cost model, is what picks the
instruction.
"""
from __future__ import annotations

import math
import time

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry.analysis import extract
from tilefoundry.schedule.facts import TileStoreFacts
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.render import emit_scaffold
from tilefoundry.schedule.select_atoms import select_atoms
from tilefoundry.target import AmxTarget
from tilefoundry.target.amx.atoms import AMX_FMA32_16x16x1_F32, NEON_FMLA_4x4x1_F32
from tilefoundry.target.facts import TARGET_FACTS

_STAGE = "core"
_ATOM_SHAPES = {
    AMX_FMA32_16x16x1_F32.name: AMX_FMA32_16x16x1_F32.shape_mnk,
    NEON_FMLA_4x4x1_F32.name: NEON_FMLA_4x4x1_F32.shape_mnk,
}
_KERNELS = {
    "mlp": qwen3.mlp,
    "self_attention": qwen3.self_attention,
    "tiled_mlp": qwen3.tiled_mlp,
}


def _run(name):
    """The CLI's own four stages, timed."""
    target = AmxTarget()
    spent = {}
    at = time.perf_counter()
    tg = extract(_KERNELS[name])
    spent["extract"], at = time.perf_counter() - at, time.perf_counter()
    tg = build_schedule_tree(tg)
    spent["build_schedule_tree"], at = time.perf_counter() - at, time.perf_counter()
    solved = select_atoms(tg, target=target, stage=_STAGE)
    spent["select_atoms"], at = time.perf_counter() - at, time.perf_counter()
    rendered = emit_scaffold(solved)
    spent["emit_scaffold"] = time.perf_counter() - at
    print(f"\n=== {name} ===")
    print("  " + "  ".join(f"{k} {v:.3f}s" for k, v in spent.items()))
    return target, tg, solved, rendered


def _happens_before(tree: "isl.schedule") -> "isl.union_map":
    """Which statement instance the schedule runs before which: two instances
    are ordered exactly when their schedule points are, so composing the
    schedule map with itself under isl's lexicographic order gives the whole
    relation in one call."""
    schedule_map = tree.get_map()
    return schedule_map.lex_lt_union_map(schedule_map)


@pytest.fixture(scope="module", params=sorted(_KERNELS))
def run(request):
    return request.param, *_run(request.param)


def test_the_schedule_respects_every_dependence(run):
    """Property 1 -- legality, checked by isl over the schedule that was built
    rather than trusted because a scheduler produced it: every dependence has
    to be one of the pairs the schedule orders."""
    name, _, tg, solved, _ = run
    assert not tg.deps.is_empty(), f"{name}: nothing to order"
    assert tg.deps.is_subset(_happens_before(solved.tree)), name


def test_reversing_the_dependences_makes_the_same_schedule_illegal(run):
    """The legality check above only means something if it can fail: the same
    schedule against the reversed dependences has to be rejected."""
    name, _, tg, solved, _ = run
    reversed_deps = tg.deps.reverse()
    assert not reversed_deps.is_subset(_happens_before(solved.tree)), name


def test_every_statement_is_granularised_or_recorded_as_uncovered(run):
    """Property 3 -- completeness. An op the catalogue does not cover is a
    recorded absence, never a silent one and never a whole-run failure."""
    name, _, tg, solved, _ = run
    statements = solved.decisions["statements"]
    assert set(statements) == {unit.name for unit in tg.units}
    for stmt_name, stmt in statements.items():
        picked, candidates = stmt["atom"], stmt["candidates"]
        assert (picked in candidates) if candidates else (picked is None), (
            f"{name}/{stmt_name}: picked {picked!r} of {candidates}"
        )


def test_the_picked_atom_granularises_its_own_tile(run):
    """Property 4 -- granularity. The atom's extent has to divide the tile on
    the dimensions it covers, which are the trailing ones: an MNK atom says
    nothing about the batch dimensions a batched matmul iterates outside them."""
    name, _, _, solved, _ = run
    checked = 0
    for stmt_name, stmt in solved.decisions["statements"].items():
        if stmt["atom"] is None:
            continue
        shape = _ATOM_SHAPES[stmt["atom"]]
        trailing = stmt["tile"][-len(shape):]
        assert len(trailing) == len(shape), f"{name}/{stmt_name}: {stmt['tile']} vs {shape}"
        for axis, (size, extent) in enumerate(zip(trailing, shape)):
            assert size % extent == 0, f"{name}/{stmt_name}: axis {axis}, {size} % {extent}"
        checked += 1
    assert checked, f"{name}: no atom was picked, so granularity proves nothing"


def _buffer_ceilings(tg) -> dict[str, int]:
    """Per buffer, the bytes the whole kernel touches of it -- the bounding box
    of every access's range. One tile can never hold more than that."""
    maps: list["isl.map"] = []
    tg.reads.union(tg.writes).foreach_map(maps.append)
    boxes: dict[str, int] = {}
    for m in maps:
        buf = m.get_tuple_name(isl.dim_type.OUT)
        rng = m.range()
        span = 1
        for pos in range(rng.dim(isl.dim_type.SET)):
            lo, hi = rng.dim_min_val(pos), rng.dim_max_val(pos)
            if not (lo.is_int() and hi.is_int()):
                return {}  # a parametric buffer has no static ceiling
            span *= int(str(hi)) - int(str(lo)) + 1
        elem = math.ceil(tg.buffer_dtypes[buf].bit_width / 8)
        boxes[buf] = max(boxes.get(buf, 0), span * elem)
    return boxes


def test_each_hole_reports_its_working_set_against_the_level_store(run):
    """Property 2 -- capacity. The picked atom fits its own storage level by
    construction (that is what made it a candidate), so what is left to report
    is the hole's whole working set against the store a core-level tile lives
    in. It is recorded, not enforced: over capacity is a worse schedule, not an
    absent one -- but it does have to be a *number*, so each buffer is also
    checked against the most of it the whole kernel ever touches."""
    name, target, tg, solved, _ = run
    capacity = TARGET_FACTS.project(target, TileStoreFacts, _STAGE).tile_capacity_bytes
    assert solved.decisions["capacity_bytes"] == capacity
    ceilings = _buffer_ceilings(tg)
    for stmt_name, stmt in solved.decisions["statements"].items():
        held = stmt["footprint_bytes"]
        assert held, f"{name}/{stmt_name}: no buffer accounted"
        total = sum(held.values())
        assert stmt["fits_capacity"] == (total <= capacity), f"{name}/{stmt_name}"
        for buf, held_bytes in held.items():
            ceiling = ceilings.get(buf)
            if ceiling is None:
                continue
            # A ring of N slots really does occupy N copies, so that is the one
            # multiplier the charge is allowed above what the buffer holds.
            slots = solved.ring.get(buf, 1)
            assert held_bytes <= ceiling * slots, (
                f"{name}/{stmt_name}/{buf}: charged {held_bytes} B for {slots} slot(s) "
                f"of a buffer the whole kernel only touches {ceiling} B of"
            )


def test_the_scaffold_has_one_hole_per_statement(run):
    """What the authoring agent is handed: a loop nest with one hole per
    statement, each naming the op it stands for and the buffers it touches."""
    name, _, tg, solved, rendered = run
    skeleton, swimlane, holes = rendered
    assert len(holes) == len(tg.units), name
    assert {hole.name for hole in holes} == set(skeleton.holes)
    assert skeleton.text.strip() and swimlane.text.strip()
    for hole in holes:
        assert hole.output.tensor_name
        assert hole.coords


def test_only_the_loop_tiled_mlp_reaches_the_register_resident_atom():
    """The pair that justifies the whole path: one value, two ways of writing
    it. ``mlp``'s matmuls accumulate a 64x6144 f32 block, which no register
    file holds, so they land on the cache-streaming atom; ``tiled_mlp``'s
    accumulate 32x32, which is exactly the Z file, so they land on AMX."""
    picked = {}
    for name in ("mlp", "tiled_mlp"):
        _, _, solved, _ = _run(name)
        picked[name] = {
            stmt["atom"]
            for stmt_name, stmt in solved.decisions["statements"].items()
            if stmt_name.startswith("MM")
        }
    print(f"\nmlp {picked['mlp']}  tiled_mlp {picked['tiled_mlp']}")

    assert picked["mlp"] == {NEON_FMLA_4x4x1_F32.name}
    assert picked["tiled_mlp"] == {AMX_FMA32_16x16x1_F32.name}
