"""RMSNorm tests: typeinfer relaxation + CPU-interpreter evaluate parity.

Typeinfer (``hir.RMSNorm``):
``rms_norm`` was previously restricted to rank-2 ``x`` with
``x.dtype == weight.dtype``. Qwen3 shapes (``[1, 1, 2048]`` ``bf16``
input + ``[2048]`` ``f32`` weight) require rank-N + dtype-mismatch
acceptance. This file locks the relaxed contract.
"""

from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.dsl import DimVar
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.types import DType, TensorType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

# ---------------------------------------------------------------------------
# Typeinfer: rank-N input + dtype-mismatch (x bf16 / weight f32) accepted;
# output keeps x's full shape and dtype. Rank-0 x / rank-2 weight / last-dim
# mismatch rejected.
# ---------------------------------------------------------------------------

_CTX_LEN = DimVar("CTX_LEN", 1, 4097)
_RMS = RMSNorm(eps=1e-6)
_PARTIAL_MESH = make_mesh((4,))

CASES = [
    TypeInferCase(
        "rank3_bf16_input_f32_weight",
        _RMS,
        (make_tensor_type((1, 1, 2048), DType.bf16), make_tensor_type((2048,), DType.f32)),
        make_tensor_type((1, 1, 2048), DType.bf16),
    ),
    # dynamic batch dim (DimVar arithmetic) flows through verbatim.
    TypeInferCase(
        "dim_arithmetic_batch_survives",
        _RMS,
        (make_tensor_type((1, _CTX_LEN + 1, 2048), DType.bf16), make_tensor_type((2048,), DType.f32)),
        make_tensor_type((1, _CTX_LEN + 1, 2048), DType.bf16),
    ),
    TypeInferCase(
        "rank0_x_rejected",
        _RMS,
        (TensorType.scalar(DType.bf16), make_tensor_type((2048,), DType.f32)),
        ExpectedError(match="x must be rank ≥ 1"),
    ),
    TypeInferCase(
        "rank2_weight_rejected",
        _RMS,
        (make_tensor_type((4, 2048), DType.bf16), make_tensor_type((1, 2048), DType.f32)),
        ExpectedError(match="weight must be rank-1"),
    ),
    TypeInferCase(
        "last_dim_mismatch_rejected",
        _RMS,
        (make_tensor_type((4, 2048), DType.bf16), make_tensor_type((1024,), DType.f32)),
        ExpectedError(match="last dim"),
    ),
    # rms_norm normalizes across an axis (non-monotonic); no reduction commutes.
    TypeInferCase(
        "partial_input_rejected",
        _RMS,
        (
            make_shard_tensor_type(
                (4, 2048), DType.bf16, mesh=_PARTIAL_MESH, attrs=(Partial("sum"),)
            ),
            make_tensor_type((2048,), DType.f32),
        ),
        ExpectedError(match="x carries Partial"),
    ),
    TypeInferCase(
        "partial_weight_rejected",
        _RMS,
        (
            make_tensor_type((4, 2048), DType.bf16),
            make_shard_tensor_type(
                (2048,), DType.f32, mesh=_PARTIAL_MESH, attrs=(Partial("sum"),)
            ),
        ),
        ExpectedError(match="weight carries Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rms_norm_typeinfer(case):
    run_typeinfer_case(case)


def test_rms_norm_evaluate():
    torch.manual_seed(0)
    _nx, _nw = torch.randn(2, 8), torch.randn(8)
    _nref = _nx * torch.rsqrt(_nx.pow(2).mean(-1, keepdim=True) + 1e-6) * _nw
    run_eval_case(EvalCase("", RMSNorm(eps=1e-6), (_nx, _nw), _nref, atol=1e-5))
