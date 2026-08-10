"""Pin ``extract`` behavior for authored ``GridRegionExpr`` loops.

Loop axes prefix enclosed statements only; nested loops order outermost first.
A carried buffer creates a distance-one dependence that scheduling must order.
``DimVar`` extents become isl parameters. Data-dependent gathers fail closed
instead of claiming a known slice. Small loops keep expected delta sets
hand-transcribable; the corpus Analyze witness covers real tiled kernels.
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
    x: Tensor[(8, 4), "f32"],
    y: Tensor[(4,), "f32"],
    idx: Tensor[(), "i32"],
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
    """Self deltas.

    ``statement``'s own dependence distances, tuple name dropped so the
    expected set reads as plain coordinates.
    """
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
    """The carry is one buffer: the ``add`` reads and writes ``o``.

    The ``add`` reads and writes ``o``, so isl reports ``[1, 0, 0]``: iteration
    ``i`` needs what ``i - 1`` wrote. The loop-invariant ``mul`` keeps its 2-d
    domain while ``add`` gains a leading loop axis. The schedule must order that
    carry strictly.
    """
    tg = extract(carry_loop)
    doms = _domains(tg)
    assert sorted(doms) == ["Binary0", "Binary1"]

    assert doms["Binary0"].dim(isl.dim_type.SET) == 2
    assert _extent(doms["Binary0"], 0) == (0, 7)
    assert doms["Binary1"].dim(isl.dim_type.SET) == 3
    assert _extent(doms["Binary1"], 0) == (0, 5)
    assert _extent(doms["Binary1"], 1) == (0, 7)
    assert _extent(doms["Binary1"], 2) == (0, 3)

    assert _writer_of(tg, "o") == "Binary1"
    assert "-> o[" in str(tg.reads), tg.reads
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [1, 0, 0] }"))

    assert tg.parallel_dims["Binary1"] == (False, True, True)
    assert tg.parallel_dims["Binary0"] == (True, True)

    assert _violations(tg).is_empty()


def test_nested_loops_contribute_one_dimension_each():
    """Two axes, innermost last: the carry advances by one inner step.

    Two axes, innermost last: the carry advances by one inner step, and
    wraps to the next outer step from the last inner one.
    """
    tg = extract(nested_carry)
    dom = _domains(tg)["Binary1"]
    assert dom.dim(isl.dim_type.SET) == 4
    assert _extent(dom, 0) == (0, 3)
    assert _extent(dom, 1) == (0, 1)
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("{ [0, 1, 0, 0]; [1, -1, 0, 0] }"))
    assert tg.parallel_dims["Binary1"] == (False, False, True, True)


def test_dynamic_extent_becomes_an_isl_parameter():
    """A loop whose trip count is only known at the call.

    A loop whose trip count is only known at the call: the axis is bounded by
    the parameter itself, and the carry distance is still one step of it.
    """
    tg = extract(dyn_carry)
    assert tg.params == {"seq": SEQ}
    dom = _domains(tg)["Binary1"]
    assert "0 <= i0 < seq" in str(dom), dom
    assert _self_deltas(tg, "Binary1").is_equal(isl.set("[seq] -> { [1, 0, 0] : 4 <= seq <= 63 }"))


def test_data_dependent_gather_fails_closed():
    """Test data dependent gather fails closed.

    A gather whose index is a value, not the loop's own induction variable,
    has no affine access map -- so extraction refuses rather than inventing one.
    """
    with pytest.raises(ExtractError, match="not an enclosing loop's induction variable"):
        extract(data_gather)
