"""RMSNorm's rejected operands.

``rms_norm`` is the most-used op in the corpus, and every decoder Reference runs
it at production shapes -- rank-3 ``bf16`` input against a rank-1 ``f32`` weight,
dynamic batch dim included -- so the accepted contract and the value oracle are
witnessed there. What a real model never builds is what this file keeps: a
weight of the wrong rank, a last dim that disagrees with it, and a
``Partial``-carrying operand (rms_norm normalizes across an axis, so no
reduction commutes).
"""

from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_RMS = RMSNorm(eps=1e-6)
_PARTIAL_MESH = make_mesh((4,))

CASES = [
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
    # x stands for weight too: one Partial check, reported by operand name.
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
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rms_norm_typeinfer(case):
    run_typeinfer_case(case)
