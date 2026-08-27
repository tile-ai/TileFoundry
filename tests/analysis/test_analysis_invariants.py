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

from tilefoundry.analysis.movement import (
    _bytes_for,
    _reached_bytes,
    call_traffic,
)
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- op names resolved dynamically
from tilefoundry.inspection.values import (
    ENTRIES,
    ENTRY,
    FIELD,
    FIELDS,
    PAIR,
    PER_UNIT,
    TRIPS,
    Prose,
)
from tilefoundry.ir.core import (
    Call,
    Constant,
    TotalAndPerUnit,
    TripInterval,
    Tuple,
    Var,
)
from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPattern
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.slice import Slice as SliceOp
from tilefoundry.ir.types import (
    DType,
    TensorType,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
    tensor_bytes,
)
from tilefoundry.ir.types.shard import Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split as ShardSplit
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    access_relation_registry,
    coordinates_of,
    index_set,
    relation_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes, TypeInferContext

B, S, H, D = 1, 5, 2, 3


HQ, HKV, HEAD_DIM, MAX_POS = 16, 8, 128, 8


def test_every_callable_op_states_its_coordinates_exactly_once() -> None:
    """One canonical relation per Op, enumerated rather than listed by hand.

    Where an Op reads and writes is stated once, so type inference, the
    polyhedral model, the loop footprint and the movement half read one answer.
    The set is taken from the Op registry itself, so an Op added to the surface
    joins this without anybody adding it here. The other dialect is not a Call
    target of these analyses and an Op a test registers is not part of the
    surface at all; both are left out by where they come from, not by name.
    """
    callable_ops = {
        schema.op_class
        for schema in iter_schemas()
        if schema.dialect == "tf"
        and schema.op_class is not None
        and schema.op_class.__module__.startswith("tilefoundry.")
    }
    assert len(callable_ops) > 40, "the surface got smaller than this gate expects"
    missing = sorted(
        op.__name__ for op in callable_ops if access_relation_registry.lookup(op) is None
    )
    assert not missing, f"callable ops with no access relation: {missing}"
    stated = set(access_relation_registry._map)
    assert stated == callable_ops, sorted(
        op.__name__ for op in stated ^ callable_ops
    )


def test_an_op_with_no_registered_relation_has_no_fallback() -> None:
    """Which is why the gate above enumerates instead of listing.

    An Op that states no coordinates is refused by name rather than given a
    default, because a default would be a second answer about where an Op reads
    and would be wrong for whichever Op it was invented for. Every Op the
    surface can call states one, so this asks with an Op the surface cannot.
    """

    class Unstated(Op):
        x = ParamDef(kind="input", pattern=TensorPattern)

    held = make_tensor_type((4,), DType.f32)
    call = Call(type=held, target=Unstated(), args=(Var(type=held, name="x"),))
    with pytest.raises(ValueError, match="Unstated states no access relations"):
        coordinates_of(call, TypeInferContext())













def test_a_boundary_reaching_past_its_operand_is_held_to_what_it_was_handed()  -> None:
    """A relation may be written past its value; a projected one never reaches there.

    An insert reads its update at the coordinate the window shifted back to, and
    for the coordinates before the window that is a negative one. Every
    projected boundary is held to the positions this participant was given, so
    what it reaches is inside them and which iterations are its own follows from
    that rather than from a read nobody could perform.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)
    destination = make_shard_tensor_type(
        (8,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    update = make_shard_tensor_type(
        (4,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    call = Call(
        type=destination,
        target=InsertSlice(),
        args=(
            Var(type=destination, name="dst"),
            Var(type=update, name="update"),
            Constant(type=make_tensor_type((), DType.i64), value=2),
        ),
    )
    ctx = CostContext(level="cta", topologies=(cta,))

    stated = coordinates_of(call, ctx)
    reads = relation_of(stated.inputs[1].pattern)
    assert not reads.intersect_range(
        isl.set("{ [c0] : c0 < 0 }")
    ).is_empty(), "the window's own read runs before its operand begins"

    relations = relations_of(call, ctx)
    held = (
        *(ctx.local_type_of(arg) for arg in call.args),
        ctx.local_type_of(call),
    )
    for boundary, view in zip((*relations.inputs, *relations.outputs), held, strict=True):
        reached = relation_of(boundary.pattern).range()
        if not isinstance(view, TensorType) or reached.is_empty():
            continue
        box = index_set(tuple(view.shape))
        assert box is not None and reached.is_subset(box), (
            f"a boundary reached {reached} outside the {tuple(view.shape)} it was given"
        )
    assert relation_of(relations.inputs[1].pattern).domain().is_equal(
        isl.set("{ [d0] : 2 <= d0 <= 3 }")
    ), "so the iterations left are the ones whose read this participant holds"





def _assert_shape(text: str, declared: object) -> None:
    """One value keeps the shape its declared type renders in.

    Which is also how the separator ladder is checked: an inner value holding an
    outer separator would not match the shape its own type renders in.
    """
    if declared is int:
        assert re.fullmatch(r"-?\d+", text), (text, declared)
    elif declared is Prose:
        assert re.fullmatch(r'"(?:[^"\\]|\\.)*"', text), (text, declared)
    elif declared is str:
        assert FIELD not in text and FIELDS not in text, (text, declared)
    elif declared is TrafficBytes:
        assert re.fullmatch(rf"r-?\d+{re.escape(PAIR)}w-?\d+", text), text
    elif declared is TripInterval:
        assert re.fullmatch(rf"\[[^{ENTRIES}]+{ENTRIES}[^{ENTRIES}]+\)(\{TRIPS}\d+)?", text), text
    elif get_origin(declared) is TotalAndPerUnit:
        (inner,) = get_args(declared)
        total, separator, per_unit = text.partition(PER_UNIT)
        assert separator == PER_UNIT, text
        _assert_shape(total, inner)
        _assert_shape(per_unit, inner)
    elif get_origin(declared) is dict:
        key_type, value_type = get_args(declared)
        for entry in text.split(ENTRIES):
            key, separator, value = entry.partition(ENTRY)
            assert separator == ENTRY, entry
            _assert_shape(key, key_type)
            _assert_shape(value, value_type)
    elif get_origin(declared) is tuple:
        (item_type,) = (arg for arg in get_args(declared) if arg is not Ellipsis)
        for item in text.split(ENTRIES):
            _assert_shape(item, item_type)
    else:
        raise AssertionError(f"{declared} has no rendering: {text}")





def test_a_reached_leaf_is_charged_at_its_own_level_and_the_others_are_not() -> None:
    """One tuple, two levels, and one boundary that only reached one of them.

    An operand's Type names every level its leaves live at, so grouping the
    amount by the Type charges the leaf nobody reached as well as the one that
    was -- twice the traffic, at a level this occurrence never touched. The
    reached leaf's bytes and the reached leaf's level have to stay together.
    """
    source = Var(type=make_tensor_type((1024, 2048)), name="source")
    narrow = make_tensor_type((), DType.i32, storage=StorageKind.GMEM)
    wide = make_tensor_type((), DType.i64, storage=StorageKind.RMEM)
    starts = Tuple(
        type=TupleType(fields=(narrow, wide)),
        elements=(Constant(type=narrow, value=0), Constant(type=wide, value=0)),
    )
    call = Call(
        type=make_tensor_type((256, 2048)),
        target=SliceOp(sizes=(256, 2048), strides=(1, 1)),
        args=(source, starts),
    )
    honest = access_relation_registry.lookup(SliceOp)

    def reads_the_second_number(one, ctx) -> AccessRelations:
        """A window that reads the second of its numbers and not the first."""
        relations = honest(one, ctx)
        held = relation_of(relations.inputs[1].pattern)
        return AccessRelations(
            inputs=(
                relations.inputs[0],
                BoundaryRelation(
                    AffineAccess(held.intersect_range(isl.set("{ [l] : l = 1 }")))
                ),
            ),
            outputs=relations.outputs,
        )

    def measured():
        return call_traffic(call, CostContext(), CostContext())

    both = measured()
    assert both.operands == (TrafficBytes(), TrafficBytes(read=12), TrafficBytes())
    assert both.whole == (
        ("gmem", TrafficBytes(read=4)),
        ("rmem", TrafficBytes(read=8)),
    ), "reading both numbers is one charge at each of their levels"

    access_relation_registry._map[SliceOp] = reads_the_second_number
    try:
        one = measured()
    finally:
        access_relation_registry._map[SliceOp] = honest
    assert measured() == both, "the honest relation was not put back"

    assert one.operands == (TrafficBytes(), TrafficBytes(read=8), TrafficBytes()), (
        "the second number is eight bytes wide"
    )
    assert one.whole == (("rmem", TrafficBytes(read=8)),), (
        "and it lives at rmem, so gmem was not touched at all"
    )
    assert one.per_unit == one.whole

    written = AccessRelations(
        inputs=(),
        outputs=(
            BoundaryRelation(
                AffineAccess(isl.map("{ [d0] -> [c0] : c0 = d0 and 0 <= d0 < 2 }"))
            ),
            BoundaryRelation(
                AffineAccess(isl.map("{ [d0] -> [c0] : c0 = 0 and 0 <= d0 < 2 }"))
            ),
        ),
    )
    result = TupleType(
        fields=(
            make_tensor_type((2,), DType.i32, storage=StorageKind.GMEM),
            make_tensor_type((2,), DType.i64, storage=StorageKind.RMEM),
        )
    )
    asked = tuple(
        (field_, boundary.pattern)
        for field_, boundary in zip(result.fields, written.outputs, strict=True)
    )
    assert _reached_bytes(asked, None) == (16, {"gmem": 8, "rmem": 8}), (
        "a field written in part owes that part, at that field's own level"
    )


def test_a_value_of_bits_is_charged_the_bytes_it_takes_up() -> None:
    """A boolean is a bit, and part of a byte is still a byte to fetch.

    Counting a packed value as a share of the whole rounds down, so one bool
    would cost nothing and nine of them one byte instead of the two they are
    read in. The width of an element is what says this, rounded up, and a value
    whose width nobody states is refused rather than guessed at.
    """
    mask = make_tensor_type((512,), DType.bool)
    assert tensor_bytes(mask) == 64

    assert [_bytes_for(mask, count) for count in (0, 1, 8, 9, 512)] == [0, 1, 1, 2, 64]

    packed = make_tensor_type((512,), DType.f4e2m1)
    assert [_bytes_for(packed, count) for count in (0, 1, 2, 3, 512)] == [0, 1, 1, 2, 256]

    held = make_tensor_type((512,), DType.f32)
    assert _bytes_for(held, 512) == tensor_bytes(held)
    assert _bytes_for(held, 1) == 4
    assert _bytes_for(held, None) is None
