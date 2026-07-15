"""Quant typeinfer: returns (quantized values, per-group f32 scales); the last
dim must be divisible by the group size."""
from __future__ import annotations

import pytest

from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TupleType
from tilefoundry.ir.types.shard.shard_layout import Partial
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)

_BF = DType.bf16
_FP8 = DType.fp8e4m3
_GMEM = StorageKind.GMEM


def _g(shape, dtype):
    return ten(shape, dtype, storage=_GMEM)


CASES = [
    TypeInferCase(
        "rank2_per_token_group_128",
        Quant(),
        (_g((1, 2048), _BF),),
        TupleType(fields=(_g((1, 2048), _FP8), _g((1, 16), DType.f32))),
    ),
    TypeInferCase(
        "rank3_attn_path",
        Quant(),
        (_g((1, 1, 4096), _BF),),
        TupleType(fields=(_g((1, 1, 4096), _FP8), _g((1, 1, 32), DType.f32))),
    ),
    TypeInferCase(
        "custom_group_size",
        Quant(group=64),
        (_g((1, 256), _BF),),
        TupleType(fields=(_g((1, 256), _FP8), _g((1, 4), DType.f32))),
    ),
    TypeInferCase(
        "indivisible_last_dim",
        Quant(),
        (_g((1, 100), _BF),),
        ExpectedError(match="not divisible by group", exc=TypeError),
    ),
    TypeInferCase(
        "rank0",
        Quant(),
        (_g((), _BF),),
        ExpectedError(match="at least rank-1", exc=TypeError),
    ),
    # per-group amax normalization does not commute with any reduction.
    TypeInferCase(
        "partial_sum_errors",
        Quant(),
        (sharded((1, 2048), (Partial("sum"),), mesh((4,)), dtype=_BF, storage=_GMEM),),
        ExpectedError(match="Partial input on x is unsound", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_quant_typeinfer(case):
    run_typeinfer_case(case)


def test_quant_default_attrs():
    op = Quant()
    assert op.scheme == "per_token_group"
    assert op.group == 128
    assert op.target_dtype is _FP8
