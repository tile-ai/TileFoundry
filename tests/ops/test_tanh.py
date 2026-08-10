"""Tanh typeinfer + Partial(R) commutation, and what it evaluates to."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.tanh import Tanh
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Tanh()
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))

CASES = [
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase("partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Tanh")),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_tanh_typeinfer(case):
    run_typeinfer_case(case)


_VALUES = torch.tensor(
    [-8.0, -1.0, -0.25, -0.03125, -0.0078125, 0.0, 0.0078125, 0.03125, 1.0, 8.0],
    dtype=torch.bfloat16,
)

EVAL_CASES = [
    EvalCase(
        "bf16_matches_torch_tanh",
        Tanh(),
        (_VALUES,),
        torch.tanh(_VALUES),
        atol=0,
        rtol=0,
    ),
    EvalCase(
        "f32_matches_torch_tanh",
        Tanh(),
        (_VALUES.float(),),
        torch.tanh(_VALUES.float()),
        atol=0,
        rtol=0,
    ),
]


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.name)
def test_tanh_evaluates_as_torch_tanh(case):
    """The op computes `torch.tanh`, bit for bit, at the dtype it is given."""
    run_eval_case(case)
