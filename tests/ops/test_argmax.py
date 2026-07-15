"""ArgMax typeinfer: the reduced axis is dropped and the result is i64."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_I64 = DType.i64

CASES = [
    TypeInferCase("default_axis_last", ArgMax(), (ten((1, 151936), DType.f32),), ten((1,), _I64)),
    TypeInferCase("explicit_axis", ArgMax(axis=1), (ten((4, 8, 16), DType.f32),), ten((4, 16), _I64)),
    TypeInferCase("rank1_scalar", ArgMax(), (ten((128,), DType.f32),), ten((), _I64)),
    TypeInferCase(
        "axis_out_of_range",
        ArgMax(axis=3),
        (ten((4,), DType.f32),),
        ExpectedError(match="out of range", exc=TypeError),
    ),
    # The winning index cannot be recovered from a per-device partial value.
    TypeInferCase(
        "partial_max_errors",
        ArgMax(),
        (sharded((1, 128), (Partial("max"),), mesh((4,))),),
        ExpectedError(match="Partial input on x is unsound", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_argmax_typeinfer(case):
    run_typeinfer_case(case)
