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

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Partial

_RMS = RMSNorm(eps=1e-06)
_PARTIAL_MESH = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "rank2_weight_rejected",
        _RMS,
        (make_tensor_type((4, 2048), DType.bf16), make_tensor_type((1, 2048), DType.f32)),
        error=ExpectedError(match="weight must be rank-1"),
    ),
    OpCase(
        "last_dim_mismatch_rejected",
        _RMS,
        (make_tensor_type((4, 2048), DType.bf16), make_tensor_type((1024,), DType.f32)),
        error=ExpectedError(match="last dim"),
    ),
    OpCase(
        "partial_input_rejected",
        _RMS,
        (
            make_shard_tensor_type(
                (4, 2048), DType.bf16, mesh=_PARTIAL_MESH, attrs=(Partial("sum"),)
            ),
            make_tensor_type((2048,), DType.f32),
        ),
        error=ExpectedError(match="x carries Partial"),
    ),
)
test_op_case = op_case_test(CASES)
