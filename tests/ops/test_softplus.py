"""Softplus evaluator value oracle + Partial(R) commutation typeinfer."""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.types.shard.shard_layout import Partial
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded

_OP = Softplus()
_M = mesh((4,))
_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)

PARTIAL_CASES = [
    # softplus is monotone increasing: commutes with max/min, not sum.
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase(
        "partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Softplus")
    ),
]


@pytest.mark.parametrize("case", PARTIAL_CASES, ids=lambda c: c.name)
def test_softplus_typeinfer_partial(case):
    run_typeinfer_case(case)


def test_softplus_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("softplus", Softplus(), (x,), torch.nn.functional.softplus(x), atol=1e-6))
