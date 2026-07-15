"""ReLU typeinfer + Partial(R) commutation."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = ReLU()
_M = mesh((4,))
_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)

CASES = [
    TypeInferCase("passthrough", _OP, (ten((4, 8), DType.f32),), ten((4, 8), DType.f32)),
    # relu is monotone increasing: commutes with max/min, not sum.
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase(
        "partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="ReLU")
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_relu_typeinfer(case):
    run_typeinfer_case(case)
