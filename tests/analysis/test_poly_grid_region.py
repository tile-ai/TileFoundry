"""``extract`` over a ``GridRegionExpr`` body -- an authored ``for ... in
tile(...)`` loop, which used to be rejected outright.

What the model has to say, and what each test pins:

- the loop axis is the *outermost* dimension of every statement it encloses,
  and only of those (a loop-invariant value stays out of the loop);
- a carried arg is one buffer read at iteration ``i`` and written at
  ``i - 1``, so ``deps`` carries a distance-1 dependence along that axis --
  the fact the ring depth downstream is derived from;
- nested loops each contribute their own dimension, innermost last;
- a ``DimVar`` extent becomes an isl parameter (as a dynamic tensor axis
  already does) and a ``step`` a stride constraint;
- ``gather(x, i, axis=a)`` on the loop's own induction variable folds into
  the consumer's access map as that axis, so consecutive iterations address
  different slices; a data-dependent gather still fails closed.

The real kernel is ``qwen3.tiled_mlp`` (numerically checked against the
untiled ``mlp`` in ``tests/models/qwen3_1_7b``); the small synthetic loops
above it keep the expected sets hand-checkable.
"""
from __future__ import annotations

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tests.models.qwen3_1_7b.decoder_layer import (
    MB,
    MT,
    NB_HID,
    NB_INT,
    NK_HID,
    NK_INT,
    NT,
)
from tests.schedule.test_kernel_schedule import _lex_nonpositive
from tilefoundry import func
from tilefoundry.analysis import extract
from tilefoundry.analysis.poly import ExtractError
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- op names resolved dynamically
from tilefoundry.schedule.kernel_schedule import build_schedule_tree

SEQ = DimVar("seq", 4, 64)


@func
def carry_loop(x: Tensor[(8, 4), "f32"], y: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = mul(x, y)
    for i in tile(6):
        o = add(o, x)
    return o


@func
def nested_carry(x: Tensor[(8, 4), "f32"], y: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = mul(x, y)
    for r in tile(4):
        for c in tile(2):
            o = add(o, x)
    return o


@func
def dyn_carry(x: Tensor[(SEQ, 4), "f32"], y: Tensor[(SEQ, 4), "f32"]) -> Tensor[(SEQ, 4), "f32"]:
    o = mul(x, y)
    for i in tile(SEQ):
        o = add(o, x)
    return o


@func
def stepped_carry(x: Tensor[(8, 4), "f32"], y: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = mul(x, y)
    for i in range(1, 9, 3):
        o = add(o, x)
    return o


@func
def row_gather(x: Tensor[(8, 4), "f32"], y: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    o = mul(y, y)
    for i in tile(8):
        o = add(o, gather(x, i, axis=0))
    return o


@func
def data_gather(
    x: Tensor[(8, 4), "f32"], y: Tensor[(4,), "f32"], idx: Tensor[(), "i32"],
) -> Tensor[(4,), "f32"]:
    o = mul(y, y)
    for i in tile(8):
        o = add(o, gather(x, idx, axis=0))
    return o


def _domains(tg) -> dict[str, "isl.set"]:
    sets: list["isl.set"] = []
    tg.domain.foreach_set(sets.append)
    return {s.get_tuple_name(): s for s in sets}


def _extent(s: "isl.set", pos: int) -> tuple[int, int]:
    return int(s.dim_min_val(pos).num_si()), int(s.dim_max_val(pos).num_si())


def _maps(um: "isl.union_map") -> list["isl.map"]:
    out: list["isl.map"] = []
    um.foreach_map(out.append)
    return out


def _writer_of(tg, buffer: str) -> str:
    """The statement that writes ``buffer``."""
    names = {
        m.get_tuple_name(isl.dim_type.IN)
        for m in _maps(tg.writes)
        if m.get_tuple_name(isl.dim_type.OUT) == buffer
    }
    assert len(names) == 1, f"{buffer}: expected one writer, got {sorted(names)}"
    return next(iter(names))


def _self_deltas(tg, statement: str) -> "isl.set":
    """``statement``'s own dependence distances, tuple name dropped so the
    expected set reads as plain coordinates."""
    own = _domains(tg)[statement].to_union_set()
    pieces: list["isl.set"] = []
    tg.deps.intersect_domain(own).intersect_range(own).deltas().foreach_set(pieces.append)
    assert len(pieces) == 1, f"{statement}: expected one delta piece, got {pieces}"
    return pieces[0].reset_tuple_id()


def _violations(tg) -> "isl.union_set":
    """The dependence deltas ``tg.tree`` does not order strictly (same check
    ``tests/schedule/test_kernel_schedule.py`` makes)."""
    sched = tg.tree.get_map()
    timed = tg.deps.apply_domain(sched).apply_range(sched)
    assert not timed.is_empty(), "every dependence must survive into time space"
    return _lex_nonpositive(timed.deltas())


# ── the loop axis, and who gets one ───────────────────────────────────────


def test_loop_axis_is_the_outermost_dimension_of_enclosed_statements_only():
    """``o = mul(x, y)`` is loop-invariant and keeps its bare 2-d domain; the
    in-loop ``add`` gains the loop axis in front of it."""
    tg = extract(carry_loop)
    doms = _domains(tg)
    assert sorted(doms) == ["Binary0", "Binary1"]

    assert _extent(doms["Binary0"], 0) == (0, 7)  # invariant: [8, 4], no loop axis
    assert doms["Binary0"].dim(isl.dim_type.SET) == 2

    assert doms["Binary1"].dim(isl.dim_type.SET) == 3
    assert _extent(doms["Binary1"], 0) == (0, 5)  # tile(6)
    assert _extent(doms["Binary1"], 1) == (0, 7)
    assert _extent(doms["Binary1"], 2) == (0, 3)


def test_carried_arg_is_a_distance_one_dependence_on_the_loop_axis():
    """The carry is one buffer: the ``add`` reads and writes ``o``, so isl's
    own flow analysis reports exactly ``[1, 0, 0]`` -- iteration ``i`` needs
    what ``i - 1`` wrote."""
    tg = extract(carry_loop)
    assert _writer_of(tg, "o") == "Binary1"
    assert "-> o[" in str(tg.reads), tg.reads
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [1, 0, 0] }"))
    # ... and that axis is therefore not parallel, while the elementwise ones are.
    assert tg.parallel_dims["Binary1"] == (False, True, True)
    assert tg.parallel_dims["Binary0"] == (True, True)


def test_nested_loops_contribute_one_dimension_each():
    """Two axes, innermost last: the carry advances by one inner step, and
    wraps to the next outer step from the last inner one."""
    tg = extract(nested_carry)
    dom = _domains(tg)["Binary1"]
    assert dom.dim(isl.dim_type.SET) == 4
    assert _extent(dom, 0) == (0, 3)  # tile(4), outer
    assert _extent(dom, 1) == (0, 1)  # tile(2), inner
    assert _self_deltas(tg, "Binary1").is_equal(
        isl.set("{ [0, 1, 0, 0]; [1, -1, 0, 0] }")
    )
    assert tg.parallel_dims["Binary1"] == (False, False, True, True)


def test_dynamic_extent_becomes_an_isl_parameter():
    tg = extract(dyn_carry)
    assert tg.params == {"seq": SEQ}
    dom = _domains(tg)["Binary1"]
    assert "0 <= i0 < seq" in str(dom), dom
    assert _self_deltas(tg, "Binary1").is_equal(
        isl.set("[seq] -> { [1, 0, 0] : 4 <= seq <= 63 }")
    )


def test_step_and_start_appear_on_the_loop_axis():
    """``range(1, 9, 3)`` -> the axis takes the raw induction values 1, 4, 7
    (a stride constraint, not a normalised trip counter), so the carry
    distance is one *step*."""
    tg = extract(stepped_carry)
    dom = _domains(tg)["Binary1"]
    assert _extent(dom, 0) == (1, 7)
    assert dom.reset_tuple_id().is_subset(isl.set("{ [i0, d0, d1] : (i0 - 1) mod 3 = 0 }"))
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [3, 0, 0] }"))


# ── loop-index addressing ─────────────────────────────────────────────────


def test_loop_index_gather_folds_into_the_access_map():
    """``gather(x, i, axis=0)`` is a view, not a statement: the loop axis
    lands at the gathered axis of ``x``, so iteration ``i`` reads row ``i``
    and the rows carry no dependence between iterations."""
    tg = extract(row_gather)
    assert [type(u.op.target).__name__ for u in tg.units] == ["Binary", "Binary"]
    reads = {str(m) for m in _maps(tg.reads)}
    assert any("-> x[i0, d0]" in m for m in reads), reads
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [1, 0] }"))


def test_data_dependent_gather_fails_closed():
    with pytest.raises(ExtractError, match="not an enclosing loop's induction variable"):
        extract(data_gather)


# ── the real kernel ───────────────────────────────────────────────────────


def test_tiled_mlp_loop_axes_and_block_shapes():
    """Every in-loop matmul of ``tiled_mlp`` is a K-step over a
    ``[MT, KT] @ [KT, NT]`` block pair batched over (token block, column
    block), with the K axis of the *loop* in front: the gate/up walk is
    ``NK_HID`` steps, the down walk ``NK_INT``."""
    tg = extract(qwen3.tiled_mlp)
    doms = _domains(tg)

    # The three accumulators are the loop carries; each is written by the
    # `add` inside its own loop.
    gate, up, out = (_writer_of(tg, buf) for buf in ("gate_z", "up_z", "out_z"))
    for name, steps, blocks in (
        (gate, NK_HID, NB_INT), (up, NK_HID, NB_INT), (out, NK_INT, NB_HID),
    ):
        dom = doms[name]
        assert dom.dim(isl.dim_type.SET) == 5
        assert _extent(dom, 0) == (0, steps - 1)
        assert [_extent(dom, i) for i in range(1, 5)] == [
            (0, MB - 1), (0, blocks - 1), (0, MT - 1), (0, NT - 1)
        ]

    matmuls = [u.name for u in tg.units if type(u.op.target).__name__ == "MatMul"]
    assert len(matmuls) == 3
    for name in matmuls:
        dom = doms[name]
        assert dom.dim(isl.dim_type.SET) == 6  # [k step, mb, nb, m, n, k]
        assert _extent(dom, 3) == (0, MT - 1)
        assert _extent(dom, 4) == (0, NT - 1)
        assert _extent(dom, 5) == (0, 63)  # KT

    # RMSNorm is loop-invariant: it stays outside both walks.
    assert doms["RN"].dim(isl.dim_type.SET) == 2


def test_tiled_mlp_carries_are_distance_one_and_schedule_legally():
    """The three accumulator carries each show up as a distance-1 dependence
    on their own loop axis, and ``build_schedule_tree`` orders every
    dependence of the whole graph strictly."""
    tg = extract(qwen3.tiled_mlp)
    for buf in ("gate_z", "up_z", "out_z"):
        name = _writer_of(tg, buf)
        assert _self_deltas(tg, name).is_equal(isl.set("{ [1, 0, 0, 0, 0] }")), buf
        assert tg.parallel_dims[name] == (False, True, True, True, True)

    tg = build_schedule_tree(tg)
    assert tg.domain.is_subset(tg.tree.get_map().domain())
    assert _violations(tg).is_empty()
