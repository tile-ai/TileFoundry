"""Sigmoid evaluator value oracle + Partial(R) commutation typeinfer."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Sigmoid()
_M = mesh((4,))
_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)

PARTIAL_CASES = [
    # sigmoid is monotone increasing: commutes with max/min, not sum.
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase(
        "partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Sigmoid")
    ),
]


@pytest.mark.parametrize("case", PARTIAL_CASES, ids=lambda c: c.name)
def test_sigmoid_typeinfer_partial(case):
    run_typeinfer_case(case)


def test_sigmoid_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("sigmoid", Sigmoid(), (x,), torch.sigmoid(x), atol=1e-6))
