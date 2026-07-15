"""Clamp typeinfer + Partial(R) commutation."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Clamp(min_val=-1.0, max_val=1.0)
_M = mesh((4,))
_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)

CASES = [
    TypeInferCase("passthrough", _OP, (ten((4, 8), DType.f32),), ten((4, 8), DType.f32)),
    # clamp is monotone non-decreasing: commutes with max/min, not sum.
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase(
        "partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Clamp")
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_clamp_typeinfer(case):
    run_typeinfer_case(case)
