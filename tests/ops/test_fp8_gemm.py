"""FP8GEMM typeinfer + Partial(R) commutation.

Same weight-replication linear family as MatMul: a pre-existing
``Partial(sum)`` on ``lhs``/``rhs`` propagates; ``max``/``min`` do not (GEMM
does not preserve order).
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, ten
from tilefoundry.ir.hir.nn.fp8_gemm import FP8GEMM
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout

_OP = FP8GEMM(block=128, out_dtype=DType.bf16)
_M = mesh((4,))
_FP8 = DType.fp8e4m3
_F32 = DType.f32


def _sharded(shape, attrs, *, dtype) -> TensorType:
    return TensorType(
        shape=shape,
        dtype=dtype,
        layout=ShardLayout(layout=Layout(shape=shape, strides=None), attrs=attrs, mesh=_M),
        storage="gmem",
    )


_LHS = ten((8, 128), _FP8)
_RHS = ten((128, 16), _FP8)
_LHS_S = ten((8, 1), _F32)
_RHS_S = ten((1, 16), _F32)

_LHS_PSUM = _sharded((8, 128), (Partial("sum"),), dtype=_FP8)
_LHS_PMAX = _sharded((8, 128), (Partial("max"),), dtype=_FP8)
_RHS_PSUM = _sharded((128, 16), (Partial("sum"),), dtype=_FP8)
_RHS_PMAX = _sharded((128, 16), (Partial("max"),), dtype=_FP8)

CASES = [
    TypeInferCase(
        "partial_sum_lhs_passes", _OP, (_LHS_PSUM, _LHS_S, _RHS, _RHS_S),
        ten((8, 16), DType.bf16, layout=_LHS_PSUM.layout),
    ),
    TypeInferCase(
        "partial_sum_rhs_passes", _OP, (_LHS, _LHS_S, _RHS_PSUM, _RHS_S),
        ten((8, 16), DType.bf16, layout=_LHS.layout),
    ),
    TypeInferCase(
        "partial_max_lhs_errors", _OP, (_LHS_PMAX, _LHS_S, _RHS, _RHS_S),
        ExpectedError(match="FP8GEMM"),
    ),
    TypeInferCase(
        "partial_max_rhs_errors", _OP, (_LHS, _LHS_S, _RHS_PMAX, _RHS_S),
        ExpectedError(match="FP8GEMM"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_fp8_gemm_typeinfer_partial(case):
    run_typeinfer_case(case)
