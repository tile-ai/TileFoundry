"""LayerNorm trailing-shape value, affine, and shard contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, Split, split_target_axes

_OP = LayerNorm(axis=-1, eps=1e-5)
_F = DType.f32
_M = make_mesh((4,))
_X = make_tensor_type((4, 8), _F)
_W = make_tensor_type((8,), _F)
_B = make_tensor_type((8,), _F)
_W_PSUM = make_shard_tensor_type((8,), mesh=_M, attrs=(Partial("sum"),))

CASES = [
    TypeInferCase("passthrough", _OP, (_X, _W, _B), _X),
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Partial("sum"),)), _W, _B),
        ExpectedError(match="LayerNorm"),
    ),
    TypeInferCase(
        "partial_weight_errors",
        _OP,
        (_X, _W_PSUM, _B),
        ExpectedError(match="weight.*Partial.*mesh axis 0"),
    ),
    TypeInferCase(
        "weight_broadcast_shape_rejected",
        LayerNorm(axis=1, eps=1e-5),
        (
            make_tensor_type((2, 3, 4), _F),
            make_tensor_type((1, 4), _F),
            make_tensor_type((3, 4), _F),
        ),
        ExpectedError(match=r"weight shape \(1, 4\).*x\.shape\[1:\].*\(3, 4\)"),
    ),
    TypeInferCase(
        "bias_broadcast_shape_rejected",
        LayerNorm(axis=-2, eps=1e-5),
        (
            make_tensor_type((2, 3, 4), _F),
            make_tensor_type((3, 4), _F),
            make_tensor_type((1, 4), _F),
        ),
        ExpectedError(match=r"bias shape \(1, 4\).*x\.shape\[1:\].*\(3, 4\)"),
    ),
    TypeInferCase(
        "affine_dtypes_must_match",
        _OP,
        (_X, make_tensor_type((8,), DType.f16), _B),
        ExpectedError(match="weight dtype.*bias dtype"),
    ),
    TypeInferCase(
        "f32_input_rejects_f16_affine",
        _OP,
        (_X, make_tensor_type((8,), DType.f16), make_tensor_type((8,), DType.f16)),
        ExpectedError(match="affine dtype.*must match x dtype"),
    ),
    TypeInferCase(
        "normalized_x_split_rejected",
        LayerNorm(axis=1, eps=1e-5),
        (
            make_shard_tensor_type((8, 4), mesh=_M, attrs=(Split(1),)),
            make_tensor_type((4,)),
            make_tensor_type((4,)),
        ),
        ExpectedError(match=r"x normalized axis 1.*Split-sharded.*Reshard"),
    ),
    TypeInferCase(
        "normalized_weight_split_rejected",
        _OP,
        (_X, make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),)), _B),
        ExpectedError(match=r"weight normalized axis 0.*Split-sharded.*Reshard"),
    ),
    TypeInferCase(
        "normalized_bias_split_rejected",
        _OP,
        (_X, _W, make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))),
        ExpectedError(match=r"bias normalized axis 0.*Split-sharded.*Reshard"),
    ),
    TypeInferCase(
        "axis_out_of_range",
        LayerNorm(axis=2, eps=1e-5),
        (_X, _W, _B),
        ExpectedError(match="axis 2 out of range for rank 2"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layer_norm_typeinfer(case):
    run_typeinfer_case(case)


def test_layer_norm_preserves_a_prefix_split() -> None:
    x = make_shard_tensor_type((8, 3, 4), mesh=_M, attrs=(Split(0),))

    result = infer_call(
        LayerNorm(axis=1, eps=1e-5),
        x,
        make_tensor_type((3, 4)),
        make_tensor_type((3, 4)),
    )

    assert isinstance(result.layout, ShardLayout)
    assert split_target_axes(result.layout, result.shape) == (0,)


@pytest.mark.parametrize(
    ("x_dtype", "affine_dtype"),
    [
        pytest.param(torch.float32, torch.float32, id="common_f32"),
        pytest.param(torch.float16, torch.float16, id="common_f16"),
        pytest.param(torch.float16, torch.float32, id="f16_f32_affine"),
        pytest.param(torch.bfloat16, torch.float32, id="bf16_f32_affine"),
    ],
)
def test_layer_norm_matches_multi_axis_torch_reference(
    x_dtype, affine_dtype
) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=x_dtype)
    weight = torch.randn(3, 4, dtype=affine_dtype)
    bias = torch.randn(3, 4, dtype=affine_dtype)
    tolerance = {
        torch.float32: 1e-5,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }[x_dtype]

    run_eval_case(
        EvalCase(
            "multi_axis_suffix",
            LayerNorm(axis=1, eps=1e-5),
            (x, weight, bias),
            F.layer_norm(x, (3, 4), weight, bias, 1e-5),
            atol=tolerance,
            rtol=tolerance,
        )
    )
