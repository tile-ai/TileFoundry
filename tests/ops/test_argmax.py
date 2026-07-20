"""ArgMax typeinfer: the reduced axis is dropped and the result is i64."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_I64 = DType.i64

CASES = [
    TypeInferCase("default_axis_last", ArgMax(), (make_tensor_type((1, 151936), DType.f32),), make_tensor_type((1,), _I64)),
    TypeInferCase("explicit_axis", ArgMax(axis=1), (make_tensor_type((4, 8, 16), DType.f32),), make_tensor_type((4, 16), _I64)),
    TypeInferCase("rank1_scalar", ArgMax(), (make_tensor_type((128,), DType.f32),), make_tensor_type((), _I64)),
    TypeInferCase(
        "axis_out_of_range",
        ArgMax(axis=3),
        (make_tensor_type((4,), DType.f32),),
        ExpectedError(match="out of range"),
    ),
    TypeInferCase(
        "partial_input_rejected",
        ArgMax(),
        (make_shard_tensor_type((4, 256), mesh=make_mesh((4,)), attrs=(Partial("max"),)),),
        ExpectedError(match="x carries Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_argmax_typeinfer(case):
    run_typeinfer_case(case)
