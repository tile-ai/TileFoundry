"""LayerNorm trailing-shape value, affine, and shard contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from tests.core.ops.ir.op_case import ExpectedError, OpCase, infer_call, op_case_test
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    ShardLayout,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Partial, Split, split_target_axes

_OP = LayerNorm(axis=-1, eps=1e-05)
_F = DType.f32
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_X = make_tensor_type((4, 8), _F)
_W = make_tensor_type((8,), _F)
_B = make_tensor_type((8,), _F)
_W_PSUM = make_shard_tensor_type((8,), mesh=_M, attrs=(Partial("sum"),))


def test_layer_norm_preserves_prefix_split() -> None:
    x = make_shard_tensor_type((8, 3, 4), mesh=_M, attrs=(Split(0),))
    result = infer_call(
        LayerNorm(axis=1, eps=1e-05), x, make_tensor_type((3, 4)), make_tensor_type((3, 4))
    )
    assert isinstance(result.layout, ShardLayout)
    assert split_target_axes(result.layout, result.shape) == (0,)


def _make_layer_norm_matches_multi_axis_torch_reference(x_dtype, affine_dtype) -> OpCase:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=x_dtype)
    weight = torch.randn(3, 4, dtype=affine_dtype)
    bias = torch.randn(3, 4, dtype=affine_dtype)
    tolerance = {torch.float32: 1e-05, torch.float16: 0.001, torch.bfloat16: 0.01}[x_dtype]
    return OpCase(
        "multi_axis_suffix",
        LayerNorm(axis=1, eps=1e-05),
        (x, weight, bias),
        value=F.layer_norm(x, (3, 4), weight, bias, 1e-05),
        atol=tolerance,
        rtol=tolerance,
    )


CASES = (
    *(
        replace(_make_layer_norm_matches_multi_axis_torch_reference(*_param.values), name=_param.id)
        for _param in [
            pytest.param(torch.float32, torch.float32, id="common_f32"),
            pytest.param(torch.float16, torch.float16, id="common_f16"),
            pytest.param(torch.float16, torch.float32, id="f16_f32_affine"),
            pytest.param(torch.bfloat16, torch.float32, id="bf16_f32_affine"),
        ]
    ),
    OpCase("passthrough", _OP, (_X, _W, _B), type=_X),
    OpCase(
        "partial_sum_errors",
        _OP,
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Partial("sum"),)), _W, _B),
        error=ExpectedError(match="LayerNorm"),
    ),
    OpCase(
        "partial_weight_errors",
        _OP,
        (_X, _W_PSUM, _B),
        error=ExpectedError(match="weight.*Partial.*mesh axis 0"),
    ),
    OpCase(
        "weight_broadcast_shape_rejected",
        LayerNorm(axis=1, eps=1e-05),
        (
            make_tensor_type((2, 3, 4), _F),
            make_tensor_type((1, 4), _F),
            make_tensor_type((3, 4), _F),
        ),
        error=ExpectedError(match="weight shape \\(1, 4\\).*x\\.shape\\[1:\\].*\\(3, 4\\)"),
    ),
    OpCase(
        "bias_broadcast_shape_rejected",
        LayerNorm(axis=-2, eps=1e-05),
        (
            make_tensor_type((2, 3, 4), _F),
            make_tensor_type((3, 4), _F),
            make_tensor_type((1, 4), _F),
        ),
        error=ExpectedError(match="bias shape \\(1, 4\\).*x\\.shape\\[1:\\].*\\(3, 4\\)"),
    ),
    OpCase(
        "affine_dtypes_must_match",
        _OP,
        (_X, make_tensor_type((8,), DType.f16), _B),
        error=ExpectedError(match="weight dtype.*bias dtype"),
    ),
    OpCase(
        "f32_input_rejects_f16_affine",
        _OP,
        (_X, make_tensor_type((8,), DType.f16), make_tensor_type((8,), DType.f16)),
        error=ExpectedError(match="affine dtype.*must match x dtype"),
    ),
    OpCase(
        "normalized_x_split_rejected",
        LayerNorm(axis=1, eps=1e-05),
        (
            make_shard_tensor_type((8, 4), mesh=_M, attrs=(Split(1),)),
            make_tensor_type((4,)),
            make_tensor_type((4,)),
        ),
        error=ExpectedError(match="x normalized axis 1.*Split-sharded.*Reshard"),
    ),
    OpCase(
        "normalized_weight_split_rejected",
        _OP,
        (_X, make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),)), _B),
        error=ExpectedError(match="weight normalized axis 0.*Split-sharded.*Reshard"),
    ),
    OpCase(
        "normalized_bias_split_rejected",
        _OP,
        (_X, _W, make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))),
        error=ExpectedError(match="bias normalized axis 0.*Split-sharded.*Reshard"),
    ),
    OpCase(
        "axis_out_of_range",
        LayerNorm(axis=2, eps=1e-05),
        (_X, _W, _B),
        error=ExpectedError(match="axis 2 out of range for rank 2"),
    ),
)
test_op_case = op_case_test(CASES)
