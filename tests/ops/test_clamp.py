"""Clamp typeinfer + Partial(R) commutation."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Clamp(min_val=-1.0, max_val=1.0)
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))

CASES = [
    TypeInferCase(
        "passthrough", _OP, (make_tensor_type((4, 8), DType.f32),), make_tensor_type((4, 8), DType.f32)
    ),
    # clamp is monotone non-decreasing: commutes with max/min, not sum.
    TypeInferCase("partial_max_passes", _OP, (_PMAX,), _PMAX),
    TypeInferCase(
        "partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Clamp")
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_clamp_typeinfer(case):
    run_typeinfer_case(case)
