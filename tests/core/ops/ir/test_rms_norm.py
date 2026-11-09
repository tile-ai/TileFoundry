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
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import nn

_RMS = nn.RMSNorm(eps=1e-06)
_PARTIAL_MESH = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "rank2_weight_rejected",
        _RMS,
        (types.TensorType.from_shape((4, 2048), expr.DType.bf16), types.TensorType.from_shape((1, 2048), expr.DType.f32)),
        error=ExpectedError(match="weight must be rank-1"),
    ),
    OpCase(
        "last_dim_mismatch_rejected",
        _RMS,
        (types.TensorType.from_shape((4, 2048), expr.DType.bf16), types.TensorType.from_shape((1024,), expr.DType.f32)),
        error=ExpectedError(match="last dim"),
    ),
    OpCase(
        "partial_input_rejected",
        _RMS,
        (
            utils.make_shard_tensor_type(
                (4, 2048), expr.DType.bf16, mesh=_PARTIAL_MESH, attrs=(types.Partial("sum"),)
            ),
            types.TensorType.from_shape((2048,), expr.DType.f32),
        ),
        error=ExpectedError(match="x carries Partial"),
    ),
)
test_op_case = op_case_test(CASES)
