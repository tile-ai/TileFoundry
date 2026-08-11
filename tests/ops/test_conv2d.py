"""Conv2D validation, value, shard relation, and projected cost."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, infer_call, run_typeinfer_case
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, ShardLayout, Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    Split,
    split_target_axes,
)
from tilefoundry.visitor_registry.contexts import TrafficBytes

_F = DType.f32
_OP = Conv2D(stride=(2, 1), padding=(1, 0), dilation=(1, 1), groups=2)
_X = make_tensor_type((2, 4, 7, 7), _F)
_W = make_tensor_type((6, 2, 3, 3), _F)
_BIAS = make_tensor_type((6,), _F)
_MESH = make_mesh((2,))


VALIDATION_CASES = [
    TypeInferCase(
        "rank4_input",
        _OP,
        (make_tensor_type((4, 7, 7)), _W, _BIAS),
        ExpectedError(match="rank-4 input and weight"),
    ),
    TypeInferCase(
        "stride_length",
        Conv2D(stride=(1,), padding=(0, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"stride must be length-2.*\(1,\)"),
    ),
    TypeInferCase(
        "padding_length",
        Conv2D(stride=(1, 1), padding=(0,), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"padding must be length-2.*\(0,\)"),
    ),
    TypeInferCase(
        "dilation_length",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1,), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"dilation must be length-2.*\(1,\)"),
    ),
    TypeInferCase(
        "stride_positive",
        Conv2D(stride=(0, 1), padding=(0, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"stride values must be positive.*\(0, 1\)"),
    ),
    TypeInferCase(
        "dilation_positive",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(-1, 1), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"dilation values must be positive.*\(-1, 1\)"),
    ),
    TypeInferCase(
        "padding_non_negative",
        Conv2D(stride=(1, 1), padding=(-1, 0), dilation=(1, 1), groups=2),
        (_X, _W, _BIAS),
        ExpectedError(match=r"padding values must be non-negative.*\(-1, 0\)"),
    ),
    TypeInferCase(
        "groups_positive",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=0),
        (_X, _W, _BIAS),
        ExpectedError(match="groups must be positive, got 0"),
    ),
    TypeInferCase(
        "input_channels_divisible",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=3),
        (_X, make_tensor_type((6, 1, 3, 3)), _BIAS),
        ExpectedError(match="input channels 4 must be divisible by groups 3"),
    ),
    TypeInferCase(
        "output_channels_divisible",
        _OP,
        (_X, make_tensor_type((5, 2, 3, 3)), make_tensor_type((5,))),
        ExpectedError(match="output channels 5 must be divisible by groups 2"),
    ),
    TypeInferCase(
        "weight_input_channels",
        _OP,
        (_X, make_tensor_type((6, 3, 3, 3)), _BIAS),
        ExpectedError(match="weight input-channel extent 3.*channels/groups 2"),
    ),
    TypeInferCase(
        "bias_rank",
        _OP,
        (_X, _W, make_tensor_type((1, 6))),
        ExpectedError(match=r"bias must be rank-1.*\(1, 6\)"),
    ),
    TypeInferCase(
        "bias_extent",
        _OP,
        (_X, _W, make_tensor_type((5,))),
        ExpectedError(match="bias extent 5 must equal output channels 6"),
    ),
    TypeInferCase(
        "bias_dtype",
        _OP,
        (_X, _W, make_tensor_type((6,), DType.f16)),
        ExpectedError(match="bias dtype.*must match input dtype"),
    ),
    TypeInferCase(
        "weight_dtype",
        _OP,
        (_X, make_tensor_type((6, 2, 3, 3), DType.f16), _BIAS),
        ExpectedError(match="weight dtype.*must match input dtype"),
    ),
]


@pytest.mark.parametrize("case", VALIDATION_CASES, ids=lambda case: case.name)
def test_conv2d_validates_its_operand_contract(case) -> None:
    run_typeinfer_case(case)


def test_grouped_conv2d_matches_torch_reference() -> None:
    torch.manual_seed(0)
    input_ = torch.randn(2, 4, 7, 7)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    expected = F.conv2d(
        input_, weight, bias, stride=(2, 1), padding=(1, 0), groups=2
    )
    run_eval_case(EvalCase("grouped", _OP, (input_, weight, bias), expected))


def test_conv2d_plain_layout_describes_the_result() -> None:
    result = infer_call(_OP, _X, _W, _BIAS)

    assert result.shape == (2, 6, 4, 5)
    assert result.layout == Layout(shape=result.shape, strides=(120, 20, 5, 1))


PROPAGATION_CASES = [
    pytest.param(
        make_shard_tensor_type((2, 4, 7, 7), mesh=_MESH, attrs=(Split(0),)),
        _W,
        _BIAS,
        (Split(0),),
        (0,),
        id="input_batch_split",
    ),
    pytest.param(
        _X,
        make_shard_tensor_type((6, 2, 3, 3), mesh=_MESH, attrs=(Split(0),)),
        make_shard_tensor_type((6,), mesh=_MESH, attrs=(Split(0),)),
        (Split(1),),
        (1,),
        id="weight_and_bias_output_channel_split",
    ),
]


@pytest.mark.parametrize(("input_", "weight", "bias", "attrs", "targets"), PROPAGATION_CASES)
def test_conv2d_relation_propagates_surviving_splits(
    input_, weight, bias, attrs, targets
) -> None:
    result = infer_call(_OP, input_, weight, bias)

    assert isinstance(result.layout, ShardLayout)
    assert all(result.layout != type_.layout for type_ in (input_, weight, bias))
    assert math.prod(result.layout.layout.shape) == math.prod(result.shape)
    assert result.layout.attrs == attrs
    assert split_target_axes(result.layout, result.shape) == targets


def test_conv2d_relation_accepts_exact_contraction_partial() -> None:
    op = Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    input_ = make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(Split(1),))
    weight = make_shard_tensor_type((4, 4, 3, 3), mesh=_MESH, attrs=(Split(1),))
    bias = make_shard_tensor_type((4,), mesh=_MESH, attrs=(Partial("sum"),))

    result = infer_call(op, input_, weight, bias)

    assert isinstance(result.layout, ShardLayout)
    assert result.layout.layout.shape == result.shape
    assert result.layout.attrs == (Partial("sum"),)


SHARD_ERROR_CASES = [
    TypeInferCase(
        "contraction_requires_partial_bias",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(Split(1),)),
            make_shard_tensor_type((4, 4, 3, 3), mesh=_MESH, attrs=(Split(1),)),
            make_tensor_type((4,)),
        ),
        ExpectedError(match=r"bias must carry Partial\(sum\).*Reshard"),
    ),
    TypeInferCase(
        "grouped_input_channel_relation_is_underivable",
        _OP,
        (
            make_shard_tensor_type((2, 4, 7, 7), mesh=_MESH, attrs=(Split(1),)),
            make_shard_tensor_type(
                (6, 2, 3, 3), mesh=_MESH, attrs=(Split(1),)
            ),
            make_shard_tensor_type((6,), mesh=_MESH, attrs=(Partial("sum"),)),
        ),
        ExpectedError(match=r"input 0.*non-projection.*Reshard"),
    ),
    TypeInferCase(
        "input_channel_split_requires_weight_peer",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(Split(1),)),
            make_tensor_type((4, 4, 3, 3)),
            make_shard_tensor_type((4,), mesh=_MESH, attrs=(Partial("sum"),)),
        ),
        ExpectedError(match=r"weight must carry a matching.*mesh axis 0.*Reshard"),
    ),
    TypeInferCase(
        "spatial_halo_is_underivable",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            make_shard_tensor_type((1, 4, 8, 8), mesh=_MESH, attrs=(Split(2),)),
            make_tensor_type((4, 4, 3, 3)),
            make_tensor_type((4,)),
        ),
        ExpectedError(match=r"input 0.*non-projection.*Reshard"),
    ),
    TypeInferCase(
        "translated_spatial_split_is_underivable",
        Conv2D(stride=(1, 1), padding=(4, 0), dilation=(1, 1), groups=1),
        (
            make_shard_tensor_type(
                (1, 4, 8, 8), mesh=make_mesh((4,)), attrs=(Split(2),)
            ),
            make_tensor_type((4, 4, 1, 1)),
            make_tensor_type((4,)),
        ),
        ExpectedError(match=r"input 0.*non-projection.*Reshard"),
    ),
    TypeInferCase(
        "output_channel_meshes_must_match",
        _OP,
        (
            _X,
            make_shard_tensor_type((6, 2, 3, 3), mesh=make_mesh((2,)), attrs=(Split(0),)),
            make_shard_tensor_type((6,), mesh=make_mesh((3,)), attrs=(Split(0),)),
        ),
        ExpectedError(match=r"bias \(input 2\) references a different mesh.*Reshard"),
    ),
    TypeInferCase(
        "broadcast_bias_mesh_must_match",
        _OP,
        (
            _X,
            make_shard_tensor_type(
                (6, 2, 3, 3), mesh=make_mesh((2,)), attrs=(Split(0),)
            ),
            make_shard_tensor_type(
                (6,), mesh=make_mesh((3,)), attrs=(Broadcast(),)
            ),
        ),
        ExpectedError(match=r"bias \(input 2\) references a different mesh.*Reshard"),
    ),
]


@pytest.mark.parametrize("case", SHARD_ERROR_CASES, ids=lambda case: case.name)
def test_conv2d_rejects_underivable_ownership(case) -> None:
    run_typeinfer_case(case)


_CTA = Topology("cta", 2)
_CTA_MESH = make_mesh((2,), topology=_CTA)
_INPUT_BYTES = 2 * 4 * 7 * 7 * 4
_WEIGHT_BYTES = 6 * 2 * 3 * 3 * 4
_BIAS_BYTES = 6 * 4
_OUTPUT_BYTES = 2 * 6 * 4 * 5 * 4

COST_CASES = [
    CostCase(
        "grouped_global",
        _OP,
        (_X, _W, _BIAS),
        flops={_F: 2 * 2 * 6 * 4 * 5 * 2 * 3 * 3},
        traffic=(
            TrafficBytes(read=_INPUT_BYTES),
            TrafficBytes(read=_WEIGHT_BYTES),
            TrafficBytes(read=_BIAS_BYTES),
            TrafficBytes(write=_OUTPUT_BYTES),
        ),
    ),
    CostCase(
        "grouped_cta_batch_projection",
        _OP,
        (
            make_shard_tensor_type(
                (2, 4, 7, 7), mesh=_CTA_MESH, attrs=(Split(0),)
            ),
            _W,
            _BIAS,
        ),
        flops={_F: 2 * 1 * 6 * 4 * 5 * 2 * 3 * 3},
        traffic=(
            TrafficBytes(read=_INPUT_BYTES // 2),
            TrafficBytes(read=_WEIGHT_BYTES),
            TrafficBytes(read=_BIAS_BYTES),
            TrafficBytes(write=_OUTPUT_BYTES // 2),
        ),
        level="cta",
        topologies=(_CTA,),
    ),
    CostCase(
        "aligned_contraction_cta_projection",
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        (
            make_shard_tensor_type(
                (1, 4, 8, 8), mesh=_CTA_MESH, attrs=(Split(1),)
            ),
            make_shard_tensor_type(
                (4, 4, 3, 3), mesh=_CTA_MESH, attrs=(Split(1),)
            ),
            make_shard_tensor_type(
                (4,), mesh=_CTA_MESH, attrs=(Partial("sum"),)
            ),
        ),
        flops={_F: 2 * 1 * 4 * 6 * 6 * 2 * 3 * 3},
        traffic=(
            TrafficBytes(read=1 * 2 * 8 * 8 * 4),
            TrafficBytes(read=4 * 2 * 3 * 3 * 4),
            TrafficBytes(read=4 * 4),
            TrafficBytes(write=1 * 4 * 6 * 6 * 4),
        ),
        level="cta",
        topologies=(_CTA,),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_conv2d_grouped_cost(case) -> None:
    run_cost_case(case)
