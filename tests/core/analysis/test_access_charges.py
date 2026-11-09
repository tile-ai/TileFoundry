"""The arithmetic the placed inventory cannot state about itself.

Running whole programs proves the answers agree; it does not prove any of them
is right. What is left here is what a wrong answer would look like at readable
dimensions: an Op nobody registered answered anyway, a boundary charged for
coordinates its participant never had, a leaf charged at another leaf's width or
level, and a packed value charged a share of a byte.
"""

from __future__ import annotations

import re
from typing import get_args, get_origin

import isl
import pytest

from tilefoundry import analysis, inspection
from tilefoundry.dsl import *
from tilefoundry.interfaces import access, cost, schedule
from tilefoundry.ir import core, dim, expr, pattern, types, utils
from tilefoundry.ir.hir import tensor

B, S, H, D = 1, 5, 2, 3


HQ, HKV, HEAD_DIM, MAX_POS = 16, 8, 128, 8


def test_callable_op_states_relation_once() -> None:
    """One canonical relation per Op, enumerated rather than listed by hand.

    Where an Op reads and writes is stated once, so type inference, the loop
    footprint and the movement half read one answer.
    The tf set is taken from the Op registry itself, so an Op added to the
    surface joins this without anybody adding it here. TIR instructions may
    also state relations for an HIR schedule to consume; they must be
    effect-declared instructions, but effect declaration alone does not require
    registration until an HIR consumer needs that instruction.
    """
    callable_ops = {
        schema.op_class
        for schema in expr.iter_schemas()
        if schema.dialect == "tf"
        and schema.op_class is not None
        and schema.op_class.__module__.startswith("tilefoundry.")
    }
    assert len(callable_ops) > 40, "the surface got smaller than this gate expects"
    missing = sorted(
        op.__name__ for op in callable_ops if not access.relation_registry.has(op)
    )
    assert not missing, f"callable ops with no access relation: {missing}"
    stated = set(access.relation_registry.entries)
    effect_declared_tir = {
        schema.op_class
        for schema in expr.iter_schemas()
        if schema.dialect == "T"
        and schema.op_class is not None
        and all(param.effect is not None for param in schema.signature if param.kind == "input")
    }
    tir_stated = stated - callable_ops
    assert tir_stated, "no TIR instruction states access relations for an HIR consumer"
    assert stated <= callable_ops | effect_declared_tir, sorted(
        op.__name__ for op in stated - callable_ops - effect_declared_tir
    )


def test_unstated_op_relation_refused() -> None:
    """Which is why the gate above enumerates instead of listing.

    An Op that states no coordinates is refused by name rather than given a
    default, because a default would be a second answer about where an Op reads
    and would be wrong for whichever Op it was invented for. Every Op the
    surface can call states one, so this asks with an Op the surface cannot.
    One that states them in any other form than one relation per argument and
    then per result is refused the same way: before its Type, a tuple of them
    with something produced; once the Type is known, one per field at its rank.
    """

    class Unstated(expr.Op):
        x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())

    held = TensorType.from_shape((4,), DType.f32)
    call = expr.Call(type=held, target=Unstated(), args=(expr.Var(type=held, name="x"),))
    with pytest.raises(ValueError, match="Unstated states no access relations"):
        schedule.operand_relations(Unstated(), (held,))

    walked = access.Relation(isl.map("{ [d0] -> [d0] : 0 <= d0 < 4 }"))
    for stated, asked, message in (
        ([walked, walked], access.relations_of, "one access.Relation each, in a tuple"),
        ((walked, walked.relation), access.relations_of, "one access.Relation each, in a tuple"),
        ((walked,), access.relations_of, "then what it produces"),
        ((walked, walked, walked), access.local_relations_of, "2 output boundaries of a call with 1"),
        (
            (walked, access.Relation(isl.map("{ [d0] -> [d0, 0] : 0 <= d0 < 4 }"))),
            access.local_relations_of,
            "reads output 0 at 2 coordinates",
        ),
    ):
        access.relation_registry.register(Unstated, lambda call, ctx, stated=stated: stated)
        try:
            with pytest.raises(ValueError, match=message):
                asked(call, cost.Context())
        finally:
            del access.relation_registry.entries[Unstated]


_CTA2 = Topology("cta", 2)
_CTA2_MESH = Mesh((_CTA2,), Layout((2,), (1,)), ("c",))
_I64 = TensorType.from_shape((), DType.i64)
_CTA4, _THREAD2 = Topology("cta", 4), Topology("thread", 2)


def _nested(extent: int, *, wrapped: bool) -> TensorType:
    """A thread Split under a replicated cta layer, directly or through an offset view."""
    inner = types.ShardLayout(
        Layout((extent,), (1,)),
        (types.Split(0),),
        Mesh((_THREAD2,), Layout((2,), (1,)), ("t",)),
    )
    held = ComposedLayout(inner=None, offset=1, outer=inner) if wrapped else inner
    return TensorType(
        shape=(extent,),
        dtype=DType.f32,
        layout=types.ShardLayout(held, (types.Broadcast(),), Mesh((_CTA4,), Layout((4,), (1,)), ("c",))),
        storage=StorageKind.GMEM,
    )


@pytest.mark.parametrize(
    ("destination", "update", "offsets", "window", "own", "level"),
    [
        pytest.param(
            make_shard_tensor_type((8,), mesh=_CTA2_MESH, attrs=(types.Split(0),), dtype=DType.f32),
            make_shard_tensor_type((4,), mesh=_CTA2_MESH, attrs=(types.Split(0),), dtype=DType.f32),
            expr.Constant(type=_I64, value=2),
            "{ [d0] : 2 <= d0 <= 5 }",
            "{ [d0] : 2 <= d0 <= 3 }",
            "cta",
            id="split_axis",
        ),
        *(
            pytest.param(
                _nested(8, wrapped=wrapped),
                _nested(4, wrapped=wrapped),
                expr.Constant(type=_I64, value=2),
                "{ [d0] : 2 <= d0 <= 5 }",
                "{ [d0] : 2 <= d0 <= 3 }",
                "thread",
                id=f"nested_{'through_an_offset_view' if wrapped else 'directly'}",
            )
            for wrapped in (False, True)
        ),
    ],
)
def test_insert_read_held_to_window(
    destination, update, offsets, window, own, level
) -> None:
    """An insert reads its update only inside the window.

    Projection cuts the read to the positions each participant was given, and
    its own iterations follow from that. A nested layer cuts once whether held
    directly or through an offset view.
    """
    call = expr.Call(
        type=destination,
        target=tensor.InsertSlice(),
        args=(expr.Var(type=destination, name="dst"), expr.Var(type=update, name="update"), offsets),
    )
    topologies = (_CTA4, _THREAD2) if level == "thread" else (_CTA2,)
    ctx = cost.Context(topology_level=level, topologies=topologies)

    stated = access.relations_of(call, ctx)
    reads = stated[1].relation
    assert reads.domain().is_equal(isl.set(window)), (
        "the unprojected insert reads its update across exactly the whole window"
    )
    assert reads.range().is_equal(dim.shape_to_isl_set(tuple(update.shape), {})), (
        "the unprojected insert reads exactly the update's window coordinates"
    )

    relations = access.local_relations_of(call, ctx)
    held = (
        *(ctx.local_type_of(arg) for arg in call.args),
        ctx.local_type_of(call),
    )
    for boundary, view in zip(relations, held, strict=True):
        reached = boundary.relation.range()
        if not isinstance(view, TensorType) or reached.is_empty():
            continue
        assert expr.is_literal_shape(view.shape), f"{tuple(view.shape)} is not a literal shape"
        box = dim.shape_to_isl_set(tuple(view.shape), {})
        assert reached.is_subset(box), (
            f"a boundary reached {reached} outside the {tuple(view.shape)} it was given"
        )
    assert relations[1].relation.domain().is_equal(isl.set(own)), (
        "so the iterations left are the ones whose read this participant holds"
    )


def _assert_shape(text: str, declared: object) -> None:
    """One value keeps the shape its declared type renders in.

    Which is also how the separator ladder is checked: an inner value holding an
    outer separator would not match the shape its own type renders in.
    """
    if declared is int:
        assert re.fullmatch(r"-?\d+", text), (text, declared)
    elif declared is inspection.Prose:
        assert re.fullmatch(r'"(?:[^"\\]|\\.)*"', text), (text, declared)
    elif declared is str:
        assert inspection.FIELD not in text and inspection.FIELDS not in text, (text, declared)
    elif declared is cost.TrafficBytes:
        assert re.fullmatch(rf"r-?\d+{re.escape(inspection.PAIR)}w-?\d+", text), text
    elif declared is core.TripInterval:
        assert re.fullmatch(rf"\[[^{inspection.ENTRIES}]+{inspection.ENTRIES}[^{inspection.ENTRIES}]+\)(\{inspection.TRIPS}\d+)?", text), text
    elif get_origin(declared) is core.TotalAndPerUnit:
        (inner,) = get_args(declared)
        total, separator, per_unit = text.partition(inspection.PER_UNIT)
        assert separator == inspection.PER_UNIT, text
        _assert_shape(total, inner)
        _assert_shape(per_unit, inner)
    elif get_origin(declared) is dict:
        key_type, value_type = get_args(declared)
        for entry in text.split(inspection.ENTRIES):
            key, separator, value = entry.partition(inspection.ENTRY)
            assert separator == inspection.ENTRY, entry
            _assert_shape(key, key_type)
            _assert_shape(value, value_type)
    elif get_origin(declared) is tuple:
        (item_type,) = (arg for arg in get_args(declared) if arg is not Ellipsis)
        for item in text.split(inspection.ENTRIES):
            _assert_shape(item, item_type)
    else:
        raise AssertionError(f"{declared} has no rendering: {text}")


def test_reached_leaf_charged_own_level() -> None:
    """One tuple, two levels, and one boundary that only reached one of them.

    An operand's Type names every level its leaves live at, so grouping the
    amount by the Type charges the leaf nobody reached as well as the one that
    was -- twice the traffic, at a level this occurrence never touched. The
    reached leaf's bytes and the reached leaf's level have to stay together.
    """
    source = expr.Var(type=TensorType.from_shape((1024, 2048)), name="source")
    narrow = TensorType.from_shape((), DType.i32, storage=StorageKind.GMEM)
    wide = TensorType.from_shape((), DType.i64, storage=StorageKind.RMEM)
    starts = expr.Tuple(
        type=TupleType(fields=(narrow, wide)),
        elements=(expr.Constant(type=narrow, value=0), expr.Constant(type=wide, value=0)),
    )
    call = expr.Call(
        type=TensorType.from_shape((256, 2048)),
        target=tensor.Slice(sizes=(256, 2048), strides=(1, 1)),
        args=(source, starts),
    )
    honest = access.relation_registry.get(tensor.Slice, None)

    def reads_the_second_number(one, ctx) -> tuple[access.Relation, ...]:
        """A window that reads the second of its numbers and not the first."""
        source, numbers, *result = honest(one, ctx)
        held = numbers.relation.intersect_range(isl.set("{ [l] : l = 1 }"))
        return (source, access.Relation(held), *result)

    def measured():
        return analysis.call_traffic(call, cost.Context(), {"cta": cost.Context()})

    both = measured()
    assert both.operands == (cost.TrafficBytes(), cost.TrafficBytes(read=12), cost.TrafficBytes())
    assert (
        both.traffic.storage.of("gmem").total,
        both.traffic.storage.of("rmem").total,
    ) == (
        cost.TrafficBytes(read=4),
        cost.TrafficBytes(read=8),
    ), "reading both numbers is one charge at each of their levels"

    access.relation_registry.entries[tensor.Slice] = reads_the_second_number
    try:
        one = measured()
    finally:
        access.relation_registry.entries[tensor.Slice] = honest
    assert measured() == both, "the honest relation was not put back"

    assert one.operands == (cost.TrafficBytes(), cost.TrafficBytes(read=8), cost.TrafficBytes()), (
        "the second number is eight bytes wide"
    )
    assert one.traffic.storage.of("rmem").total == cost.TrafficBytes(read=8), (
        "and it lives at rmem, so gmem was not touched at all"
    )
    assert one.traffic.storage.of("gmem") is None, "gmem was not touched at all"
    assert one.traffic.storage.of("rmem").per_unit == (cost.TrafficBytes(read=8),), (
        "one CTA is the only unit, so its share is the whole"
    )

    written = (
        access.Relation(isl.map("{ [d0] -> [c0] : c0 = d0 and 0 <= d0 < 2 }")),
        access.Relation(isl.map("{ [d0] -> [c0] : c0 = 0 and 0 <= d0 < 2 }")),
    )
    result = TupleType(
        fields=(
            TensorType.from_shape((2,), DType.i32, storage=StorageKind.GMEM),
            TensorType.from_shape((2,), DType.i64, storage=StorageKind.RMEM),
        )
    )
    asked = tuple(zip(result.fields, written, strict=True))
    assert analysis.reached_bytes(asked, None) == (16, {"gmem": 8, "rmem": 8}), (
        "a field written in part owes that part, at that field's own level"
    )


def test_bit_values_charge_whole_bytes() -> None:
    """A boolean is a bit, and part of a byte is still a byte to fetch.

    Counting a packed value as a share of the whole rounds down, so one bool
    would cost nothing and nine of them one byte instead of the two they are
    read in. The width of an element is what says this, rounded up, and a value
    whose width nobody states is refused rather than guessed at.
    """
    mask = TensorType.from_shape((512,), DType.bool)
    assert utils.tensor_bytes(mask) == 64

    assert [analysis.bytes_for(mask, count) for count in (0, 1, 8, 9, 512)] == [0, 1, 1, 2, 64]

    packed = TensorType.from_shape((512,), DType.f4e2m1)
    assert [analysis.bytes_for(packed, count) for count in (0, 1, 2, 3, 512)] == [0, 1, 1, 2, 256]

    held = TensorType.from_shape((512,), DType.f32)
    assert analysis.bytes_for(held, 512) == utils.tensor_bytes(held)
    assert analysis.bytes_for(held, 1) == 4
    assert analysis.bytes_for(held, None) is None
