"""``extract`` over a ``GridRegionExpr`` body -- an authored ``for ... in
tile(...)`` loop, which used to be rejected outright.

What the model has to say, and what each test pins:

- the loop axis is the *outermost* dimension of every statement it encloses,
  and only of those (a loop-invariant value stays out of the loop);
- a carried arg is one buffer read at iteration ``i`` and written at
  ``i - 1``, so ``deps`` carries a distance-1 dependence along that axis --
  the fact the ring depth downstream is derived from, and the fact the
  private schedule tree then has to order strictly;
- nested loops each contribute their own dimension, innermost last;
- a ``DimVar`` extent becomes an isl parameter (as a dynamic tensor axis
  already does);
- ``gather(x, i, axis=a)`` on a data-dependent index fails closed rather than
  producing a map that claims to know which slice is read.

The loops here are small and synthetic on purpose: the expected delta sets are
hand-transcribed, which is only possible at this size. A real tiled kernel's own
loop structure is exercised by the corpus Analyze witness.
"""
from __future__ import annotations

import isl
import pytest

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
    """The dependence deltas ``build_schedule_tree`` does not order strictly."""
    sched = build_schedule_tree(tg).get_map()
    timed = tg.deps.apply_domain(sched).apply_range(sched)
    assert not timed.is_empty(), "every dependence must survive into time space"
    return _lex_nonpositive(timed.deltas())


def _lex_nonpositive(deltas: "isl.union_set") -> "isl.union_set":
    out = isl.union_set("{}")
    pieces: list = []
    deltas.foreach_set(pieces.append)
    for piece in pieces:
        rank = piece.dim(isl.dim_type.SET)
        dims = ", ".join(f"d{i}" for i in range(rank))
        positive = isl.set(
            "{ "
            + "; ".join(
                f"[{dims}] : "
                + " and ".join([*(f"d{i} = 0" for i in range(index)), f"d{index} > 0"])
                for index in range(rank)
            )
            + " }"
        )
        out = out.union(piece.subtract(positive))
    return out


def test_carried_arg_is_a_distance_one_dependence_on_the_loop_axis():
    """The carry is one buffer: the ``add`` reads and writes ``o``, so isl's
    own flow analysis reports exactly ``[1, 0, 0]`` -- iteration ``i`` needs
    what ``i - 1`` wrote.

    The loop axis belongs to the enclosed statement only: ``o = mul(x, y)`` is
    loop-invariant and keeps its bare 2-d domain, while the in-loop ``add`` gains
    the loop axis in front of it. And the schedule tree built over these facts has
    to order that carry strictly -- a distance the tree does not respect is a plan
    that runs an iteration before the one it reads from.
    """
    tg = extract(carry_loop)
    doms = _domains(tg)
    assert sorted(doms) == ["Binary0", "Binary1"]

    assert doms["Binary0"].dim(isl.dim_type.SET) == 2  # invariant: [8, 4]
    assert _extent(doms["Binary0"], 0) == (0, 7)
    assert doms["Binary1"].dim(isl.dim_type.SET) == 3
    assert _extent(doms["Binary1"], 0) == (0, 5)  # tile(6)
    assert _extent(doms["Binary1"], 1) == (0, 7)
    assert _extent(doms["Binary1"], 2) == (0, 3)

    assert _writer_of(tg, "o") == "Binary1"
    assert "-> o[" in str(tg.reads), tg.reads
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [1, 0, 0] }"))
    # ... and that axis is therefore not parallel, while the elementwise ones are.
    assert tg.parallel_dims["Binary1"] == (False, True, True)
    assert tg.parallel_dims["Binary0"] == (True, True)

    assert _violations(tg).is_empty()


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
    """A loop whose trip count is only known at the call: the axis is bounded by
    the parameter itself, and the carry distance is still one step of it."""
    tg = extract(dyn_carry)
    assert tg.params == {"seq": SEQ}
    dom = _domains(tg)["Binary1"]
    assert "0 <= i0 < seq" in str(dom), dom
    assert _self_deltas(tg, "Binary1").is_equal(
        isl.set("[seq] -> { [1, 0, 0] : 4 <= seq <= 63 }")
    )


def test_data_dependent_gather_fails_closed():
    """A gather whose index is a value, not the loop's own induction variable,
    has no affine access map -- so extraction refuses rather than inventing one."""
    with pytest.raises(ExtractError, match="not an enclosing loop's induction variable"):
        extract(data_gather)
