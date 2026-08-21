"""Cover Unary Partial algebra, low-precision propagation, and value oracles.

Cases distinguish linear, monotone, and noncommuting operation classes. FP8
type inference, f16 evaluation, ABS, and ties-to-even ROUND fill corpus gaps.

See [hir §1.3](docs/spec/hir.md#13-op).
"""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.visitor_registry.contexts import TrafficBytes

_NEG = Unary(kind=UnaryKind.NEG)
_EXP = Unary(kind=UnaryKind.EXP)
_ABS = Unary(kind=UnaryKind.ABS)
_RSQRT = Unary(kind=UnaryKind.RSQRT)
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))


CASES = [
    TypeInferCase(
        "low_precision_passthrough_fp8e4m3",
        _NEG,
        (make_tensor_type((4, 8), DType.fp8e4m3),),
        make_tensor_type((4, 8), DType.fp8e4m3),
    ),
    TypeInferCase("neg_partial_sum_passes", _NEG, (_PSUM,), _PSUM),
    TypeInferCase("neg_partial_max_errors", _NEG, (_PMAX,), ExpectedError(match="carries Partial")),
    TypeInferCase("exp_partial_max_passes", _EXP, (_PMAX,), _PMAX),
    TypeInferCase("exp_partial_sum_errors", _EXP, (_PSUM,), ExpectedError(match="carries Partial")),
    TypeInferCase("abs_partial_sum_errors", _ABS, (_PSUM,), ExpectedError(match="carries Partial")),
    TypeInferCase(
        "rsqrt_partial_max_errors", _RSQRT, (_PMAX,), ExpectedError(match="carries Partial")
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_unary_typeinfer(case):
    run_typeinfer_case(case)


ORACLES = [
    pytest.param("abs", _ABS, lambda: torch.randn(4), torch.abs, id="abs_f32"),
    pytest.param(
        "round_ties_to_even",
        Unary(kind=UnaryKind.ROUND),
        lambda: torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        torch.round,
        id="round_ties_to_even",
    ),
    pytest.param("exp", _EXP, lambda: torch.randn(4, dtype=torch.float16), torch.exp, id="exp_f16"),
    pytest.param(
        "exp",
        _EXP,
        lambda: torch.randn(4, dtype=torch.bfloat16),
        torch.exp,
        id="exp_bf16",
    ),
]


@pytest.mark.parametrize(("name", "op", "drawn", "oracle"), ORACLES)
def test_a_unary_evaluates_to_its_oracle(name, op, drawn, oracle):
    torch.manual_seed(0)
    x = drawn()

    run_eval_case(EvalCase(name, op, (x,), oracle(x)))


_SPECIAL = (
    UnaryKind.RSQRT,
    UnaryKind.EXP,
    UnaryKind.LOG,
    UnaryKind.EXP2,
    UnaryKind.LOG2,
)
_REAL = make_tensor_type((8,), DType.f32)
_WHOLE = make_tensor_type((8,), DType.i32)
_TRUTHY = make_tensor_type((8,), DType.bool)

COST_CASES = [
    CostCase(
        "exp_is_special_function_work",
        Unary(kind=UnaryKind.EXP),
        (_REAL,),
        service={"special": 8},
        traffic=(TrafficBytes(read=32), TrafficBytes(write=32)),
    ),
    CostCase(
        "neg_over_reals_is_floating_point_work",
        _NEG,
        (_REAL,),
        flops={DType.f32: 8},
        traffic=(TrafficBytes(read=32), TrafficBytes(write=32)),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_unary_charges_the_unit_that_answers_it(case):
    """An exponential is served by a different unit than a negation.

    The special-function kinds have their own published throughput, a quarter of
    the scalar one on this architecture, so counting one as a single FLOP would
    put it on the float pipe four times too fast. Negating a truth is not
    arithmetic at all, and negating a whole number is not floating point.
    """
    run_cost_case(case)
    assert DType.bool not in case.flops
