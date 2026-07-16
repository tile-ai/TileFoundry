"""Softplus evaluator value oracle + Partial(R) commutation typeinfer."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Softplus()
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))

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
