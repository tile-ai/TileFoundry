"""Cover Unary Partial algebra, low-precision propagation, and value oracles.

Cases distinguish linear, monotone, and noncommuting operation classes. FP8
type inference, f16 evaluation, ABS, and ties-to-even ROUND fill corpus gaps.

See [hir §1.3](docs/spec/hir.md#13-op).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, check_op_case, op_case_test
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import math

_NEG = math.Unary(kind=expr.UnaryKind.NEG)
_EXP = math.Unary(kind=expr.UnaryKind.EXP)
_ABS = math.Unary(kind=expr.UnaryKind.ABS)
_RSQRT = math.Unary(kind=expr.UnaryKind.RSQRT)
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_PSUM = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),))
_PMAX = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("max"),))
ORACLES = [
    pytest.param("abs", _ABS, lambda: torch.randn(4), torch.abs, id="abs_f32"),
    pytest.param(
        "round_ties_to_even",
        math.Unary(kind=expr.UnaryKind.ROUND),
        lambda: torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        torch.round,
        id="round_ties_to_even",
    ),
    pytest.param("exp", _EXP, lambda: torch.randn(4, dtype=torch.float16), torch.exp, id="exp_f16"),
    pytest.param(
        "exp", _EXP, lambda: torch.randn(4, dtype=torch.bfloat16), torch.exp, id="exp_bf16"
    ),
]


def _make_a_unary_evaluates_to_its_oracle(name, op, drawn, oracle):
    torch.manual_seed(0)
    x = drawn()
    return OpCase(name, op, (x,), value=oracle(x))


_SPECIAL = (expr.UnaryKind.RSQRT, expr.UnaryKind.EXP, expr.UnaryKind.LOG, expr.UnaryKind.EXP2, expr.UnaryKind.LOG2)
_REAL = types.TensorType.from_shape((8,), expr.DType.f32)
_WHOLE = types.TensorType.from_shape((8,), expr.DType.i32)
_TRUTHY = types.TensorType.from_shape((8,), expr.DType.bool)
COST_CASES = [
    OpCase(
        "exp_is_special_function_work",
        math.Unary(kind=expr.UnaryKind.EXP),
        (_REAL,),
        cost=Cost(ops={"special": 8}, traffic=(cost.TrafficBytes(read=32), cost.TrafficBytes(write=32))),
    ),
    OpCase(
        "neg_over_reals_is_floating_point_work",
        _NEG,
        (_REAL,),
        cost=Cost(flops={expr.DType.f32: 8}, traffic=(cost.TrafficBytes(read=32), cost.TrafficBytes(write=32))),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_unary_charges_by_unit(case):
    """An exponential is served by a different unit than a negation.

    The special-function kinds have their own published throughput, a quarter of
    the scalar one on this architecture, so counting one as a single FLOP would
    put it on the float pipe four times too fast. Negating a truth is not
    arithmetic at all, and negating a whole number is not floating point.
    """
    check_op_case(case)
    assert expr.DType.bool not in case.cost.flops


CASES = (
    *(
        replace(_make_a_unary_evaluates_to_its_oracle(*_param.values), name=_param.id)
        for _param in ORACLES
    ),
    OpCase(
        "low_precision_passthrough_fp8e4m3",
        _NEG,
        (types.TensorType.from_shape((4, 8), expr.DType.fp8e4m3),),
        type=types.TensorType.from_shape((4, 8), expr.DType.fp8e4m3),
    ),
    OpCase("neg_partial_sum_passes", _NEG, (_PSUM,), type=_PSUM),
    OpCase("neg_partial_max_errors", _NEG, (_PMAX,), error=ExpectedError(match="carries Partial")),
    OpCase("exp_partial_max_passes", _EXP, (_PMAX,), type=_PMAX),
    OpCase("exp_partial_sum_errors", _EXP, (_PSUM,), error=ExpectedError(match="carries Partial")),
    OpCase("abs_partial_sum_errors", _ABS, (_PSUM,), error=ExpectedError(match="carries Partial")),
    OpCase(
        "rsqrt_partial_max_errors", _RSQRT, (_PMAX,), error=ExpectedError(match="carries Partial")
    ),
)
test_op_case = op_case_test(CASES)
