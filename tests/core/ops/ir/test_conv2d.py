"""Conv2D validation, value, shard relation, and projected cost."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, infer_call, op_case_test
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import nn

_F = expr.DType.f32
_OP = nn.Conv2D(stride=(2, 1), padding=(1, 0), dilation=(1, 1), groups=2)
_X = types.TensorType.from_shape((2, 4, 7, 7), _F)
_W = types.TensorType.from_shape((6, 2, 3, 3), _F)
_BIAS = types.TensorType.from_shape((6,), _F)
_MESH = types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",))


def _make_grouped_conv2d_matches_torch_reference() -> OpCase:
    torch.manual_seed(0)
    input_ = torch.randn(2, 4, 7, 7)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    expected = F.conv2d(input_, weight, bias, stride=(2, 1), padding=(1, 0), groups=2)
    return OpCase("grouped", _OP, (input_, weight, bias), value=expected)


def test_conv2d_layout_describes_result() -> None:
    result = infer_call(_OP, _X, _W, _BIAS)
    assert result.shape == (2, 6, 4, 5)
    assert result.layout == types.Layout(shape=result.shape, strides=(120, 20, 5, 1))


PROPAGATION_CASES = [
    pytest.param(
        utils.make_shard_tensor_type((2, 4, 7, 7), mesh=_MESH, attrs=(types.Split(0),)),
        _W,
        _BIAS,
        (types.Split(0),),
        (0,),
        id="input_batch_split",
    ),
    pytest.param(
        _X,
        utils.make_shard_tensor_type((6, 2, 3, 3), mesh=_MESH, attrs=(types.Split(0),)),
        utils.make_shard_tensor_type((6,), mesh=_MESH, attrs=(types.Split(0),)),
        (types.Split(1),),
        (1,),
        id="weight_and_bias_output_channel_split",
    ),
]


@pytest.mark.parametrize(("input_", "weight", "bias", "attrs", "targets"), PROPAGATION_CASES)
def test_conv2d_relation_propagates_surviving_splits(input_, weight, bias, attrs, targets) -> None:
    result = infer_call(_OP, input_, weight, bias)
    assert isinstance(result.layout, types.ShardLayout)
    assert all((result.layout != type_.layout for type_ in (input_, weight, bias)))
    assert math.prod(result.layout.layout.shape) == math.prod(result.shape)
    assert result.layout.attrs == attrs
    assert types.split_target_axes(result.layout, result.shape) == targets


def test_conv2d_accepts_contraction_partial() -> None:
    op = nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    input_ = utils.make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(types.Split(1),))
    weight = utils.make_shard_tensor_type((4, 4, 3, 3), mesh=_MESH, attrs=(types.Split(1),))
    bias = utils.make_shard_tensor_type((4,), mesh=_MESH, attrs=(types.Partial("sum"),))
    result = infer_call(op, input_, weight, bias)
    assert isinstance(result.layout, types.ShardLayout)
    assert result.layout.layout.shape == result.shape
    assert result.layout.attrs == (types.Partial("sum"),)


_CTA = types.Topology("cta", 2)
_CTA_MESH = types.Mesh((_CTA,), types.Layout((2,), (1,)), ("g",))
_INPUT_BYTES = 2 * 4 * 7 * 7 * 4
_WEIGHT_BYTES = 6 * 2 * 3 * 3 * 4
_BIAS_BYTES = 6 * 4
_OUTPUT_BYTES = 2 * 6 * 4 * 5 * 4
CASES = (
    _make_grouped_conv2d_matches_torch_reference(),
    OpCase(
        "rank4_input",
        _OP,
        (types.TensorType.from_shape((4, 7, 7)), _W, _BIAS),
        error=ExpectedError(match="rank-4 input and weight"),
    ),
    OpCase(
        "stride_length",
        nn.Conv2D(stride=(1,), padding=(0, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="stride must be length-2.*\\(1,\\)"),
    ),
    OpCase(
        "padding_length",
        nn.Conv2D(stride=(1, 1), padding=(0,), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="padding must be length-2.*\\(0,\\)"),
    ),
    OpCase(
        "dilation_length",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1,), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="dilation must be length-2.*\\(1,\\)"),
    ),
    OpCase(
        "stride_positive",
        nn.Conv2D(stride=(0, 1), padding=(0, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="stride values must be positive.*\\(0, 1\\)"),
    ),
    OpCase(
        "dilation_positive",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(-1, 1), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="dilation values must be positive.*\\(-1, 1\\)"),
    ),
    OpCase(
        "padding_non_negative",
        nn.Conv2D(stride=(1, 1), padding=(-1, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        error=ExpectedError(match="padding values must be non-negative.*\\(-1, 0\\)"),
    ),
    OpCase(
        "groups_positive",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=0),
        (_X, _W, _BIAS),
        error=ExpectedError(match="groups must be positive, got 0"),
    ),
    OpCase(
        "input_channels_divisible",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=3),
        (_X, types.TensorType.from_shape((6, 1, 3, 3)), _BIAS),
        error=ExpectedError(match="input channels 4 must be divisible by groups 3"),
    ),
    OpCase(
        "output_channels_divisible",
        _OP,
        (_X, types.TensorType.from_shape((5, 2, 3, 3)), types.TensorType.from_shape((5,))),
        error=ExpectedError(match="output channels 5 must be divisible by groups 2"),
    ),
    OpCase(
        "weight_input_channels",
        _OP,
        (_X, types.TensorType.from_shape((6, 3, 3, 3)), _BIAS),
        error=ExpectedError(match="weight input-channel extent 3.*channels/groups 2"),
    ),
    OpCase(
        "bias_rank",
        _OP,
        (_X, _W, types.TensorType.from_shape((1, 6))),
        error=ExpectedError(match="bias must be rank-1.*\\(1, 6\\)"),
    ),
    OpCase(
        "bias_extent",
        _OP,
        (_X, _W, types.TensorType.from_shape((5,))),
        error=ExpectedError(match="bias extent 5 must equal output channels 6"),
    ),
    OpCase(
        "bias_dtype",
        _OP,
        (_X, _W, types.TensorType.from_shape((6,), expr.DType.f16)),
        error=ExpectedError(match="bias dtype.*must match input dtype"),
    ),
    OpCase(
        "weight_dtype",
        _OP,
        (_X, types.TensorType.from_shape((6, 2, 3, 3), expr.DType.f16), _BIAS),
        error=ExpectedError(match="weight dtype.*must match input dtype"),
    ),
    OpCase(
        "contraction_requires_partial_bias",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            utils.make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(types.Split(1),)),
            utils.make_shard_tensor_type((4, 4, 3, 3), mesh=_MESH, attrs=(types.Split(1),)),
            types.TensorType.from_shape((4,)),
        ),
        error=ExpectedError(match="bias must carry Partial\\(sum\\).*Reshard"),
    ),
    OpCase(
        "grouped_input_channel_relation_is_underivable",
        _OP,
        (
            utils.make_shard_tensor_type((2, 4, 7, 7), mesh=_MESH, attrs=(types.Split(1),)),
            utils.make_shard_tensor_type((6, 2, 3, 3), mesh=_MESH, attrs=(types.Split(1),)),
            utils.make_shard_tensor_type((6,), mesh=_MESH, attrs=(types.Partial("sum"),)),
        ),
        error=ExpectedError(match="input 0.*non-projection.*Reshard"),
    ),
    OpCase(
        "input_channel_split_requires_weight_peer",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            utils.make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(types.Split(1),)),
            types.TensorType.from_shape((4, 4, 3, 3)),
            utils.make_shard_tensor_type((4,), mesh=_MESH, attrs=(types.Partial("sum"),)),
        ),
        error=ExpectedError(match="weight must carry a matching.*mesh axis 0.*Reshard"),
    ),
    OpCase(
        "spatial_halo_is_underivable",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            utils.make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(types.Split(2),)),
            types.TensorType.from_shape((4, 4, 3, 3)),
            types.TensorType.from_shape((4,)),
        ),
        error=ExpectedError(match="input 0.*non-projection.*Reshard"),
    ),
    OpCase(
        "translated_spatial_split_is_underivable",
        nn.Conv2D(stride=(1, 1), padding=(4, 0), dilation=(1, 1), groups=1),
        (
            utils.make_shard_tensor_type(
                (1, 4, 8, 8),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Split(2),),
            ),
            types.TensorType.from_shape((4, 4, 1, 1)),
            types.TensorType.from_shape((4,)),
        ),
        error=ExpectedError(match="input 0.*non-projection.*Reshard"),
    ),
    OpCase(
        "grouped_global",
        _OP,
        (_X, _W, _BIAS),
        cost=Cost(
            flops={_F: 2 * 2 * 6 * 4 * 5 * 2 * 3 * 3},
            traffic=(
                cost.TrafficBytes(read=_INPUT_BYTES),
                cost.TrafficBytes(read=_WEIGHT_BYTES),
                cost.TrafficBytes(read=_BIAS_BYTES),
                cost.TrafficBytes(write=_OUTPUT_BYTES),
            ),
        ),
    ),
    OpCase(
        "grouped_cta_batch_projection",
        _OP,
        (utils.make_shard_tensor_type((2, 4, 7, 7), mesh=_CTA_MESH, attrs=(types.Split(0),)), _W, _BIAS),
        cost=Cost(
            flops={_F: 2 * 1 * 6 * 4 * 5 * 2 * 3 * 3},
            traffic=(
                cost.TrafficBytes(read=_INPUT_BYTES // 2),
                cost.TrafficBytes(read=_WEIGHT_BYTES),
                cost.TrafficBytes(read=_BIAS_BYTES),
                cost.TrafficBytes(write=_OUTPUT_BYTES // 2),
            ),
            topology_level="cta",
            topologies=(_CTA,),
        ),
    ),
    OpCase(
        "aligned_contraction_cta_projection",
        nn.Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            utils.make_shard_tensor_type((1, 4, 8, 8), mesh=_CTA_MESH, attrs=(types.Split(1),)),
            utils.make_shard_tensor_type((4, 4, 3, 3), mesh=_CTA_MESH, attrs=(types.Split(1),)),
            utils.make_shard_tensor_type((4,), mesh=_CTA_MESH, attrs=(types.Partial("sum"),)),
        ),
        cost=Cost(
            flops={_F: 2 * 1 * 4 * 6 * 6 * 2 * 3 * 3},
            traffic=(
                cost.TrafficBytes(read=1 * 2 * 8 * 8 * 4),
                cost.TrafficBytes(read=4 * 2 * 3 * 3 * 4),
                cost.TrafficBytes(read=4 * 4),
                cost.TrafficBytes(write=1 * 4 * 6 * 6 * 4),
            ),
            topology_level="cta",
            topologies=(_CTA,),
        ),
    ),
)
test_op_case = op_case_test(CASES)
