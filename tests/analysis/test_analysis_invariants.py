"""Compare hard decoder access relations and dependences to hand-written maps.

Real-model analysis proves coverage, not correctness. These tests pin row
reductions, floor-divided expansion, multi-output calls, and data-dependent
gathers at readable dimensions. Expected maps use semantic forms rather than
implementation output, so a formula round-trip cannot validate itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import get_args, get_origin

import isl
import pytest

from tilefoundry import func
from tilefoundry.analysis import (
    AnalysisError,
    extract,
)
from tilefoundry.analysis.movement import (
    _bytes_for,
    _movement,
    _reached_bytes,
    call_traffic,
)
from tilefoundry.dsl import Tensor
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
    TypeInferContext,
    Var,
)
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.slice import Slice as SliceOp
from tilefoundry.ir.hir.tensor.transpose import Transpose
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
    broadcast_access,
    coordinates_of,
    identity_access,
    index_set,
    linearized_view,
    reached_elements,
    relation_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor

REPEATS = 4
B, S, H, D = 1, 5, 2, 3


HQ, HKV, HEAD_DIM, MAX_POS = 16, 8, 128, 8


@func
def elementwise_pair(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    y = sigmoid(x)  # noqa: F405
    z = exp(y)  # noqa: F405
    return z


@func
def softmax_row(x: Tensor[(2, 64), "f32"]) -> Tensor[(2, 64), "f32"]:
    y = softmax(x, axis=-1)  # noqa: F405
    return y


def test_an_op_with_no_registered_relation_has_no_fallback() -> None:
    """Which is why every op a decoder reaches has to carry one.

    ``extract`` has no fallback when the relation registry has no entry. Elementwise
    and fused row reductions therefore must register relations. ``SoftMax`` is
    pinned exactly: one statement owns a row, with the reduced axis existential
    in access maps and absent from the domain.
    """
    elementwise = extract(elementwise_pair)
    assert [type(u.op.target).__name__ for u in elementwise.units] == [
        "Sigmoid",
        "Unary",
    ]

    rows = extract(softmax_row)
    assert [u.name for u in rows.units] == ["SoftMax"]
    assert rows.domain.is_equal(isl.union_set("{ SoftMax[i] : 0 <= i < 2 }"))
    assert rows.reads.is_equal(
        isl.union_map("{ SoftMax[i] -> x[i, j] : 0 <= i < 2 and 0 <= j < 64 }")
    )
    assert rows.writes.is_equal(
        isl.union_map("{ SoftMax[i] -> y[i, j] : 0 <= i < 2 and 0 <= j < 64 }")
    )

    assert "-> y[" not in str(rows.reads)






_UNTYPED = "the result Type this Call is about to be given"


@dataclass
class _NothingKnowsTheResult(TypeInferContext):
    """A context that refuses to say what one Call returns.

    Type inference asks for the coordinates in order to derive that Type, so a
    relation stated in terms of it cannot be the one type inference asks: the
    question would be its own answer. Refusing here is the only way to tell a
    handler that derived the space from its inputs from one that asked for the
    result and got it because the context was willing to infer it again.
    """

    about: object = None

    def type_of(self, expr):
        if expr is self.about:
            raise AssertionError(
                "the canonical relation asked for the result Type it is being "
                "asked in order to derive"
            )
        return super().type_of(expr)

    def local_type_of(self, expr):
        if expr is self.about:
            raise AssertionError(
                "the canonical relation asked for the result's local Type, which "
                "is a projection of an answer that does not exist yet"
            )
        return super().local_type_of(expr)


def _pre_type(target, *args) -> AccessRelations:
    """One Op's relations, asked before anything knows what it returns."""
    handler = access_relation_registry.lookup(type(target))
    assert handler is not None
    call = Call(type=_UNTYPED, target=target, args=args)
    return handler(call, _NothingKnowsTheResult(about=call))


def _walked_by(relations: AccessRelations) -> "isl.set":
    """The space every boundary of one Op is asked by, compared as a set.

    Two spellings of one space are one space, so this asks isl whether they are
    equal rather than whether they read the same.
    """
    walked = None
    for boundary in (*relations.inputs, *relations.outputs):
        own = relation_of(boundary.pattern).domain()
        if walked is None:
            walked = own
            continue
        assert walked.is_equal(own), (
            f"one Op states one space: {walked} against {own}"
        )
    assert walked is not None
    return walked


def test_one_relation_answers_before_its_result_type_exists() -> None:
    """The registry that answers analysis is the one type inference has to ask.

    A shape and a shard layout are derived from where an Op reads and writes, so
    the coordinates cannot be stated in terms of what the Op returns -- that is
    the thing being derived. Every rank here is built from the inputs and the
    attributes: a product with and without batch axes, an elementwise op at
    three ranks, a broadcast, a permutation, a reduction, and the two fixed
    tiles whose space is the instruction's own count.
    """
    f32 = DType.f32
    cases = (
        ("matmul", MatMul(), (make_tensor_type((8, 4), f32), make_tensor_type((4, 2), f32))),
        (
            "batched matmul",
            MatMul(),
            (make_tensor_type((3, 8, 4), f32), make_tensor_type((3, 4, 2), f32)),
        ),
        ("unary rank 1", Unary(kind=UnaryKind.NEG), (make_tensor_type((5,), f32),)),
        ("unary rank 3", Unary(kind=UnaryKind.NEG), (make_tensor_type((2, 3, 4), f32),)),
        (
            "broadcast",
            Binary(kind=BinaryKind.ADD),
            (make_tensor_type((4, 8), f32), make_tensor_type((8,), f32)),
        ),
        ("permutation", Transpose(perm=(1, 0)), (make_tensor_type((3, 5), f32),)),
        (
            "reduction",
            Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM),
            (make_tensor_type((2, 3, 4), f32),),
        ),
        (
            "m16n8k16 tile",
            Mma_SM80_16x8x16(dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32),
            (
                make_tensor_type((16, 16), DType.f16),
                make_tensor_type((16, 8), DType.f16),
            ),
        ),
        (
            "m64n128k16 tile",
            Wgmma_SM90_64x128x16(
                dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32
            ),
            (
                make_tensor_type((64, 16), DType.f16),
                make_tensor_type((16, 128), DType.f16),
            ),
        ),
    )
    for label, target, shapes in cases:
        args = tuple(
            Var(type=shape, name=f"operand{index}") for index, shape in enumerate(shapes)
        )
        relations = _pre_type(target, *args)
        assert len(relations.inputs) == len(args), label
        assert relations.outputs, label
        _walked_by(relations)


def test_the_shape_type_inference_answers_is_the_one_the_relation_reaches() -> None:
    """Two answers that must be one: what a Call's Type says, and where it writes.

    Type inference derives the result's extents from the output boundary, so a
    relation that reached other coordinates would infer another shape. Asserted
    where the two could disagree: a contraction drops the axis it sums, a
    reduction drops the axes it reduces, and a permutation reorders them.
    """
    f32 = DType.f32
    for target, shapes, expected in (
        (MatMul(), ((8, 4), (4, 2)), (8, 2)),
        (MatMul(), ((3, 8, 4), (3, 4, 2)), (3, 8, 2)),
        (Transpose(perm=(1, 0)), ((3, 5),), (5, 3)),
        (Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM), ((2, 3, 4),), (2, 4)),
    ):
        args = tuple(
            Var(type=make_tensor_type(shape, f32), name=f"operand{index}")
            for index, shape in enumerate(shapes)
        )
        relations = _pre_type(target, *args)
        reached = relation_of(relations.outputs[0].pattern).range()
        extents = tuple(
            int(str(reached.dim_max_val(axis))) + 1
            for axis in range(reached.dim(isl.dim_type.SET))
        )
        assert extents == expected, f"{type(target).__name__} reaches {extents}"
        inferred = TypeInferVisitor(TypeInferContext()).visit(
            Call(type=_UNTYPED, target=target, args=args)
        )
        assert tuple(inferred.shape) == expected == extents, (
            f"{type(target).__name__}: inference and the relation disagree"
        )


def _as_map(pattern) -> "isl.map":
    """One comparable carrier, whichever affine form a boundary stated."""
    return relation_of(pattern)


def _counted(relations) -> tuple[list[int], list[int]]:
    """Every boundary's amount, as its own relation reaches it: inputs, outputs."""
    return (
        [reached_elements(item.pattern) for item in relations.inputs],
        [reached_elements(item.pattern) for item in relations.outputs],
    )


def _asked(op, result, *operands, ctx=None):
    """One Call's relations, built the way a consumer builds them."""
    call = Call(
        type=result,
        target=op,
        args=tuple(
            Var(type=type_, name=f"a{index}") for index, type_ in enumerate(operands)
        ),
    )
    return relations_of(call, ctx or TypeInferContext())


_CONV_CTA = Topology("cta", 2)
_CONV_MESH = make_mesh((2,), ("c",), topology=_CONV_CTA)


def _sharded(shape, attrs, dtype=DType.f16):
    return make_shard_tensor_type(shape, mesh=_CONV_MESH, attrs=attrs, dtype=dtype)


def test_a_boundary_states_one_carrier_and_refuses_the_others() -> None:
    """One carrier, so a reader never asks which kind it was handed.

    A bare isl map or function says where it reaches and nothing about the
    values its parameters stand for, and whoever restricts it then guesses. The
    carrier that says both is the only one a boundary takes, and every helper
    that builds one hands it over already wrapped.
    """
    for raw in (isl.map("{ [d0] -> [d0] }"), isl.multi_aff("{ [d0] -> [d0] }")):
        with pytest.raises(ValueError, match="through an AffineAccess"):
            BoundaryRelation(raw)
    with pytest.raises(ValueError, match="through an AffineAccess"):
        BoundaryRelation("{ [d0] -> [d0] }")

    held = BoundaryRelation(AffineAccess(isl.map("{ [d0] -> [d0] }")))
    assert isinstance(held.pattern, AffineAccess)
    assert isinstance(identity_access(2), AffineAccess), (
        "a helper hands over the carrier, not the map inside it"
    )
    assert isinstance(broadcast_access((4, 8), (8,)), AffineAccess)
    assert isinstance(linearized_view((2, 3), (6,)), AffineAccess)
    assert AffineAccess(isl.multi_aff("{ [d0] -> [d0] }")).relation.is_equal(
        isl.map("{ [d0] -> [d0] }")
    ), "a function is the relation it is, and is kept as one"

    for op, operands in (
        (MatMul(), ((4, 8), (8, 2))),
        (Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM), ((4, 8),)),
        (Transpose(perm=(1, 0)), ((2, 4),)),
    ):
        relations = _asked(
            op,
            make_tensor_type((4, 2), DType.f32),
            *(make_tensor_type(shape, DType.f32) for shape in operands),
        )
        for boundary in (*relations.inputs, *relations.outputs):
            assert isinstance(boundary.pattern, AffineAccess)


def test_a_partial_boundary_is_narrowed_and_not_cut_away() -> None:
    """A participant's share cuts every boundary, and a partial one survives it.

    An insert states two things about its destination: the window it writes and
    the rest it keeps. Splitting that destination halves both -- the participant
    keeps its own half of the complement -- and reading the window boundary's
    silence outside its own coordinates as a restriction would cut the
    complement to nothing instead.
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

    whole = relations_of(call, TypeInferContext())
    assert _counted(whole) == ([4, 4, 1], [4])

    unit = relations_of(call, CostContext(level="cta", topologies=(cta,)))
    assert _counted(unit) == ([2, 2, 1], [2]), (
        "each participant keeps half the complement and writes half the window"
    )
    assert not relation_of(unit.inputs[0].pattern).is_empty(), (
        "the rest of the container is still there to keep"
    )


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


_KEYED = re.compile(rf'(?P<key>[\w-]+){FIELD}(?P<value>"(?:[^"\\]|\\.)*"|\S+)')


def _emitted(text: str, family: str) -> dict[str, str]:
    """One comment taken apart by the record and field layers of the ladder.

    The record layer splits outside a quoted value, which is the whole reason a
    value that holds a separator has to bracket itself. That the pairs rejoin to
    exactly what was read is how the split is held to being one.
    """
    if text.startswith(f"{family}{FIELD}"):
        return {family: text[len(family) + len(FIELD) :]}
    head, _, rest = text.partition(FIELDS)
    assert head == family, text
    keyed = list(_KEYED.finditer(rest))
    assert FIELDS.join(match.group(0) for match in keyed) == rest, rest
    return {match["key"]: match["value"] for match in keyed}


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


_PROSE_PROBE = 'l2 holds 2 B, spills; says "no" over \\'


def _reached(pattern, extent: int):
    """The coordinates a window reaches, walking every element it has."""
    domain = isl.set(f"{{ [d0] : 0 <= d0 < {extent} }}")
    return domain.apply(pattern.relation)


def test_what_a_boundary_moves_is_what_its_relation_reaches() -> None:
    """A quantity is not a second field to keep in step with a relation.

    Reaching one element from many coordinates moves it once, so a broadcast
    costs its source and not its dependences. Reaching past the coordinates an
    operand has is not reaching, so a relation stated over a bigger container
    still answers for the smaller one it was given.
    """
    walked = "0 <= d0 < 8 and 0 <= d1 < 4"
    replicated = AffineAccess(isl.map(f"{{ [d0, d1] -> [0, d1] : {walked} }}"))
    assert reached_elements(replicated, index_set((1, 4))) == 4, (
        "one row read by eight is one row moved"
    )
    beyond = AffineAccess(isl.map(f"{{ [d0, d1] -> [d0, d1] : {walked} }}"))
    assert reached_elements(beyond, index_set((2, 4))) == 8, (
        "and coordinates the operand does not have are not reached at all"
    )


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
        return call_traffic(call, CostEvaluator(CostContext()), CostEvaluator(CostContext()))

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


def test_an_evaluator_that_reports_the_wrong_operand_count_is_refused() -> None:
    """Too few and too many are the same mistake, and both have to be caught.

    Rebuilding a cost from its relations pairs each boundary with the direction
    the evaluator reported, and pairing stops at the shorter of the two -- so an
    evaluator reporting one too many would have its extra silently dropped and
    come out the right length.
    """
    call = Call(
        type=make_tensor_type((4,), DType.f32),
        target=Binary(kind=BinaryKind.ADD),
        args=(
            Var(type=make_tensor_type((4,), DType.f32), name="a"),
            Var(type=make_tensor_type((4,), DType.f32), name="b"),
        ),
    )
    for count in (2, 4):
        cost = Cost({}, tuple(TrafficBytes(read=1) for _ in range(count)))
        with pytest.raises(AnalysisError, match=f"cost reports {count} operands"):
            _movement(call, cost, CostContext(), (call.type,) * 3)


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
