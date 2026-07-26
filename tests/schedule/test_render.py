"""``emit_scaffold(TileGraph) -> (Skeleton, Swimlane,
list[HoleContract])`` -- the render step, one stage past ``extract`` ->
``build_schedule_tree`` (``test_poly_model.py`` / ``test_kernel_schedule.py``).
Reuses that exact gemm+rmsnorm HIR so the expected statement
names/coordinates (``MM[i,j,k]``, ``RN[i]``) line up 1:1 with what this
test asserts.
"""
from __future__ import annotations

import dataclasses
import re

import isl
import pytest

from tilefoundry import func
from tilefoundry.analysis import extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import build_schedule_tree, schedule_bands, tile_band
from tilefoundry.schedule.render import (
    BufferAccess,
    EmitScaffoldError,
    HoleContract,
    Skeleton,
    Swimlane,
    emit_scaffold,
)


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def _emit():
    tg = extract(gemm_rmsnorm)
    tg = build_schedule_tree(tg)
    return tg, emit_scaffold(tg)


def test_skeleton_has_holed_calls_inside_the_loop_nest():
    """The skeleton keeps isl's own loop nest (``for c0/c1/c2``, PoC 11)
    but every naked statement call is rendered as a ``HOLE_<name>(...)``
    call carrying its schedule coordinates -- never the bare
    ``MM(c0, c1, c2)`` isl's default codegen would print."""
    _tg, (skeleton, _swimlane, _contracts) = _emit()
    print("\n=== skeleton.text ===")
    print(skeleton.text)

    assert isinstance(skeleton, Skeleton)
    assert "HOLE_MM" in skeleton.text and "HOLE_RN" in skeleton.text
    assert skeleton.holes == ("HOLE_MM", "HOLE_RN")

    # isl's own loop nest is still there.
    assert "for (int c0" in skeleton.text
    assert "for (int c1" in skeleton.text
    assert "for (int c2" in skeleton.text

    # Hole calls carry /*coords*/ with the real iterator names.
    assert re.search(r"HOLE_MM\(.*?/\*coords\*/ c0, c1, c2\);", skeleton.text)
    assert re.search(r"HOLE_RN\(.*?/\*coords\*/ c0\);", skeleton.text)

    # A conservative sync placeholder rides next to every hole call.
    assert skeleton.text.count("// barrier") == 2


def test_swimlane_is_legal_mermaid_gantt():
    """A ```mermaid gantt``` fenced block, one section per statement: MM's
    16-instance domain is minimally unrolled (prologue/steady/epilogue +
    an elided-middle marker), RN's 2-instance domain is small enough to
    show whole (nothing collapsed)."""
    _tg, (_skeleton, swimlane, _contracts) = _emit()
    print("\n=== swimlane ===")
    print(swimlane.text)

    assert isinstance(swimlane, Swimlane)
    lines = swimlane.text.splitlines()
    assert lines[0] == "```mermaid"
    assert lines[1] == "gantt"
    assert lines[-1] == "```"
    assert any(line.strip().startswith("dateFormat") for line in lines)
    assert any(line.strip() == "section MM" for line in lines)
    assert any(line.strip() == "section RN" for line in lines)

    task_line = re.compile(r"^\s*\S.*:\d+, \d+d$")
    task_lines = [line for line in lines[1:-1] if task_line.match(line)]
    assert len(task_lines) > 0

    # MM's instances are collapsed (not fully unrolled)...
    assert "elided" in swimlane.text
    # ...but RN's are small enough that nothing needs to be.
    rn_section = swimlane.text.split("section RN")[1]
    assert "elided" not in rn_section


def test_hole_contracts_one_per_statement_with_op_ref_and_bufferaccesses():
    """One HoleContract per statement (MM, RN), each carrying its real
    op_ref (the HIR Call TileUnit.op, by identity), its schedule coords,
    and a BufferAccess per read/write with a real isl index_map."""
    tg, (_skeleton, _swimlane, contracts) = _emit()

    assert len(contracts) == 2
    by_name = {c.name: c for c in contracts}
    assert set(by_name) == {"HOLE_MM", "HOLE_RN"}

    op_by_stmt = {u.name: u.op for u in tg.units}

    mm = by_name["HOLE_MM"]
    assert isinstance(mm, HoleContract)
    assert mm.op_ref is op_by_stmt["MM"]
    assert mm.coords == ("c0", "c1", "c2")
    assert all(isinstance(v, BufferAccess) for v in mm.inputs)
    assert isinstance(mm.output, BufferAccess)
    # x, w (true inputs) + h (the k-reduction's RMW self-read) -- included
    # honestly rather than silently dropped, see _ordered_inputs.
    assert {v.tensor_name for v in mm.inputs} == {"x", "w", "h"}
    assert mm.output.tensor_name == "h"
    assert all(v.dtype is not None for v in mm.inputs)
    assert mm.output.dtype is not None
    for v in mm.inputs:
        assert isinstance(v.index_map, isl.map)
        assert v.index_map.get_tuple_name(isl.dim_type.IN) == "MM"

    rn = by_name["HOLE_RN"]
    assert rn.op_ref is op_by_stmt["RN"]
    assert rn.coords == ("c0",)
    assert {v.tensor_name for v in rn.inputs} == {"h", "weight"}
    assert rn.output.tensor_name == "y"
    assert rn.output.index_map.get_tuple_name(isl.dim_type.IN) == "RN"


def test_ring_mod_index_is_reserved_but_wired():
    """V1's ``build_schedule_tree()`` always leaves ``ring`` empty (mirrors
    ``test_kernel_schedule.py``'s own ``tree.ring == {}`` assertion), so the
    ``ring[buf] = N -> buf[<last coord> % N]`` rendering path has no real
    scheduler-produced input to exercise it against -- only a hand-built
    ``TileGraph`` variant, here."""
    tg = extract(gemm_rmsnorm)
    tg = build_schedule_tree(tg)
    assert tg.ring == {}

    ring_tg = dataclasses.replace(tg, ring={"h": 3})
    skeleton, _swimlane, _contracts = emit_scaffold(ring_tg)

    print("\n=== skeleton.text (ring={'h': 3}) ===")
    print(skeleton.text)

    # h is MM's accumulator (self-read + output, at MM's coords) and RN's
    # input (at RN's own coords) -- every reference to h is now indexed by
    # that statement's own last coordinate, mod the ring depth.
    assert "h[(c2) % 3]" in skeleton.text  # MM's own coords are (c0, c1, c2)
    assert "h[(c0) % 3]" in skeleton.text  # RN's own coords are (c0,)
    assert re.search(r"/\*in\*/ x, w, h\[\(c2\) % 3\], /\*out\*/ h\[\(c2\) % 3\]", skeleton.text)
    # unrelated buffers (x, w, weight, y) are unaffected.
    assert re.search(r"/\*in\*/ h\[\(c0\) % 3\], weight, /\*out\*/ y", skeleton.text)


def test_ring_index_parenthesises_a_compound_tiled_coordinate():
    """A tiled band's innermost coordinate is a sum (``c0 + c3``), and C
    binds ``%`` tighter than ``+`` -- the ring index must parenthesise it or
    it silently means ``c0 + (c3 % N)``."""
    tg = build_schedule_tree(extract(gemm_rmsnorm))
    band = schedule_bands(tg.tree)[0]
    tiled = dataclasses.replace(tg, tree=tile_band(band, (1, 2, 4)), ring={"h": 3})

    skeleton, _swimlane, _contracts = emit_scaffold(tiled)
    print("\n=== skeleton.text (tiled, ring={'h': 3}) ===")
    print(skeleton.text)

    compound = re.findall(r"h\[([^\]]+)\]", skeleton.text)
    assert compound, "the tiled skeleton must still index h by its ring"
    for index in compound:
        assert re.fullmatch(r"\([^()]+\) % 3", index), index


def test_emit_scaffold_before_schedule_raises_clear_error():
    """``tg.tree`` is ``None`` straight out of ``extract()`` -- calling
    ``emit_scaffold`` before ``build_schedule_tree(tg)`` fails closed with a message
    naming the missing step, not a confusing isl ``AttributeError``."""
    tg = extract(gemm_rmsnorm)
    assert tg.tree is None
    with pytest.raises(EmitScaffoldError, match="schedule"):
        emit_scaffold(tg)
