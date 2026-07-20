"""Quant typeinfer: returns (quantized values, per-group f32 scales); the last
dim must be divisible by the group size."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_BF = DType.bf16
_FP8 = DType.fp8e4m3

CASES = [
    TypeInferCase(
        "rank2_per_token_group_128",
        Quant(),
        (make_tensor_type((1, 2048), _BF),),
        TupleType(fields=(make_tensor_type((1, 2048), _FP8), make_tensor_type((1, 16), DType.f32))),
    ),
    TypeInferCase(
        "custom_group_size",
        Quant(group=64),
        (make_tensor_type((1, 256), _BF),),
        TupleType(fields=(make_tensor_type((1, 256), _FP8), make_tensor_type((1, 4), DType.f32))),
    ),
    TypeInferCase(
        "indivisible_last_dim",
        Quant(),
        (make_tensor_type((1, 100), _BF),),
        ExpectedError(match="not divisible by group"),
    ),
    TypeInferCase(
        "rank0",
        Quant(),
        (make_tensor_type((), _BF),),
        ExpectedError(match="at least rank-1"),
    ),
    TypeInferCase(
        "partial_input_rejected",
        Quant(),
        (make_shard_tensor_type((1, 2048), mesh=make_mesh((4,)), attrs=(Partial("max"),)),),
        ExpectedError(match="x carries Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_quant_typeinfer(case):
    run_typeinfer_case(case)
