"""Binary's shard propagation and its Partial truth table.

Binary derives its output ``ShardLayout`` from the shared shard-propagation
engine: a layout mismatch between genuinely-sharded operands is an error, not a
silent lhs pick, and output storage anchors on concrete residency.
"""

from __future__ import annotations

import pytest

from tests.core.ops.ir.op_case import (
    Cost,
    ExpectedError,
    OpCase,
    check_op_case,
    infer_call,
    op_case_test,
)
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import math

_ADD = math.Binary(kind=expr.BinaryKind.ADD)
_MUL = math.Binary(kind=expr.BinaryKind.MUL)
_SUB = math.Binary(kind=expr.BinaryKind.SUB)
_F = expr.DType.f32
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_MAB = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
_PSUM = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),))
_PMAX = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("max"),))
_BCAST = types.TensorType.from_shape((16, 8), _F)
_PSUM_AXIS0 = utils.make_shard_tensor_type((16, 8), mesh=_MAB, attrs=(types.Partial("sum"), types.Broadcast()))
_PSUM_AXIS1 = utils.make_shard_tensor_type((16, 8), mesh=_MAB, attrs=(types.Broadcast(), types.Partial("sum")))


def test_broadcast_fallback_keeps_shard_mismatch():
    narrow = (1, 1, 16, 1, 128)
    wide = (1, 1, 16, 4, 128)
    lhs = types.TensorType.from_shape(narrow, _F, layout=types.Layout(narrow))
    rhs = types.TensorType.from_shape(wide, _F, layout=types.Layout(wide))
    assert infer_call(_MUL, lhs, rhs).layout == types.Layout(wide)
    assert infer_call(_MUL, lhs, types.TensorType.from_shape(wide, _F)).layout == types.Layout(wide)
    replicated = utils.make_shard_tensor_type(narrow, mesh=_M, attrs=(types.Broadcast(),))
    replicated_out = infer_call(_MUL, replicated, rhs)
    assert replicated_out.layout.layout.shape == wide
    split_0 = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Split(0),))
    split_1 = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Split(1),))
    with pytest.raises(expr.VerifyError, match="incompatible"):
        infer_call(_ADD, split_0, split_1)


def test_lower_rank_split_right_aligns():
    """Test lower rank split right aligns.

    A lower-rank sharded operand's Split lands on the *output* axis it
    right-aligns to, whichever side carries it. Checked as which mesh axis holds
    Split on that output axis, not as the internal layout position count a valid
    derivation happens to produce.
    """
    split_1d = utils.make_shard_tensor_type((8,), mesh=_M, attrs=(types.Split(0),))
    plain_2d = types.TensorType.from_shape((4, 8), _F)
    for lhs, rhs in ((plain_2d, split_1d), (split_1d, plain_2d)):
        out = infer_call(_ADD, lhs, rhs)
        assert out.shape == (4, 8)
        assert out.layout.attrs == (types.Split(1),)


_COMPARISONS = (
    expr.BinaryKind.EQ,
    expr.BinaryKind.NE,
    expr.BinaryKind.LT,
    expr.BinaryKind.LE,
    expr.BinaryKind.GT,
    expr.BinaryKind.GE,
)
_TRUTH = (expr.BinaryKind.AND, expr.BinaryKind.OR)
_REAL = types.TensorType.from_shape((8,), expr.DType.f32)
_WHOLE = types.TensorType.from_shape((8,), expr.DType.i32)
_TRUTHY = types.TensorType.from_shape((8,), expr.DType.bool)
COST_CASES = [
    OpCase(
        "lt_over_reals_is_a_predicate",
        math.Binary(kind=expr.BinaryKind.LT),
        (_REAL, _REAL),
        cost=Cost(
            ops={"predicate": 8},
            traffic=(cost.TrafficBytes(read=32), cost.TrafficBytes(read=32), cost.TrafficBytes(write=1)),
        ),
    ),
    OpCase(
        "add_over_reals_is_floating_point_work",
        _ADD,
        (_REAL, _REAL),
        cost=Cost(
            flops={expr.DType.f32: 8},
            traffic=(cost.TrafficBytes(read=32), cost.TrafficBytes(read=32), cost.TrafficBytes(write=32)),
        ),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_binary_charges_by_work_kind(case):
    """What a Binary costs follows from what it does, not from its result dtype.

    A comparison produces booleans and asks a machine for a predicate; an add
    over whole numbers asks for integer arithmetic; only an add over reals is a
    FLOP. Reading the work off the result dtype produced `bool` FLOPs, a rate no
    target publishes, so `performance` refused the program instead of timing it.
    """
    check_op_case(case)
    assert expr.DType.bool not in case.cost.flops


CASES = (
    OpCase(
        "different_rank_broadcast",
        _ADD,
        (types.TensorType.from_shape((4, 8), _F), types.TensorType.from_shape((8,), _F)),
        type=types.TensorType.from_shape((4, 8), _F),
    ),
    OpCase(
        "empty_axis",
        _ADD,
        (types.TensorType.from_shape((1, 0, 8), _F), types.TensorType.from_shape((8,), _F)),
        type=types.TensorType.from_shape((1, 0, 8), _F),
    ),
    OpCase(
        "conflicting_concrete_storage",
        _ADD,
        (
            types.TensorType.from_shape((4, 8), _F, storage="gmem"),
            types.TensorType.from_shape((4, 8), _F, storage="rmem"),
        ),
        error=ExpectedError(match="conflicting storage"),
    ),
    OpCase(
        "unmaterialized_abstains_from_storage",
        _ADD,
        (
            types.TensorType.from_shape((4, 8), _F, storage=expr.StorageKind.UMAT),
            types.TensorType.from_shape((4, 8), _F, storage="rmem"),
        ),
        type=types.TensorType.from_shape((4, 8), _F, layout=types.Layout((4, 8), (8, 1)), storage="rmem"),
    ),
    OpCase("add_partial_sum_partial_sum_passes", _ADD, (_PSUM, _PSUM), type=_PSUM),
    OpCase(
        "add_partial_max_partial_max_errors",
        _ADD,
        (_PMAX, _PMAX),
        error=ExpectedError(match="do not commute"),
    ),
    OpCase(
        "add_partial_sum_broadcast_errors",
        _ADD,
        (_PSUM, _BCAST),
        error=ExpectedError(match="carries Partial"),
    ),
    OpCase("add_partial_max_broadcast_passes", _ADD, (_PMAX, _BCAST), type=_PMAX),
    OpCase("mul_partial_sum_broadcast_passes", _MUL, (_PSUM, _BCAST), type=_PSUM),
    OpCase(
        "mul_partial_max_broadcast_errors",
        _MUL,
        (_PMAX, _BCAST),
        error=ExpectedError(match="carries Partial"),
    ),
    OpCase(
        "sub_partial_sum_broadcast_errors",
        _SUB,
        (_PSUM, _BCAST),
        error=ExpectedError(match="carries Partial"),
    ),
    OpCase(
        "partial_sum_different_mesh_axes_errors",
        _ADD,
        (_PSUM_AXIS0, _PSUM_AXIS1),
        error=ExpectedError(match="mesh axis 0"),
    ),
)
test_op_case = op_case_test(CASES)
