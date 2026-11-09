"""Stack derives fresh layouts through its input-to-slice relation."""

from __future__ import annotations

import math

import torch

from tests.core.ops.ir.op_case import (
    Cost,
    ExpectedError,
    OpCase,
    check_op_case,
    infer_call,
    op_case_test,
)
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor


def test_stack_result_has_fresh_layout() -> None:
    source_type = types.TensorType.from_shape((2, 3), layout=types.Layout(shape=(2, 3), strides=(3, 1)))
    result = infer_call(tensor.Stack(axis=1), source_type, source_type)
    assert result.shape == (2, 2, 3)
    assert result.layout == types.Layout(shape=(2, 2, 3), strides=(6, 3, 1))
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = left + 10
    check_op_case(
        OpCase(
            "negative_axis", tensor.Stack(axis=-1), (left, right), value=torch.stack((left, right), dim=-1)
        )
    )


def test_stack_shifts_split_past_axis() -> None:
    mesh = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
    plain = types.TensorType.from_shape((2, 8))
    sharded = utils.make_shard_tensor_type((2, 8), mesh=mesh, attrs=(types.Split(1),))
    result = infer_call(tensor.Stack(axis=0), plain, sharded)
    assert isinstance(result.layout, types.ShardLayout)
    assert result.layout != sharded.layout
    assert math.prod(result.layout.layout.shape) == math.prod(result.shape)
    assert types.split_target_axes(result.layout, result.shape) == (2,)


def test_stack_carries_uniform_partial() -> None:
    partial = utils.make_shard_tensor_type(
        (2, 8),
        mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
        attrs=(types.Partial("sum"),),
    )
    result = infer_call(tensor.Stack(axis=1), partial, partial)
    assert isinstance(result.layout, types.ShardLayout)
    assert result.layout.layout.shape == result.shape
    assert result.layout.attrs == (types.Partial("sum"),)


def _make_stack_costs_each_input_read_and_the_result_write() -> OpCase:
    input_bytes = 2 * 3 * 4
    return OpCase(
        "stack",
        tensor.Stack(axis=0),
        (types.TensorType.from_shape((2, 3), expr.DType.f32),) * 2,
        cost=Cost(
            traffic=(
                cost.TrafficBytes(read=input_bytes),
                cost.TrafficBytes(read=input_bytes),
                cost.TrafficBytes(write=2 * input_bytes),
            )
        ),
    )


CASES = (
    _make_stack_costs_each_input_read_and_the_result_write(),
    OpCase(
        "incompatible_split_axes",
        tensor.Stack(axis=0),
        (
            utils.make_shard_tensor_type(
                (8, 8),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Split(0),),
            ),
            utils.make_shard_tensor_type(
                (8, 8),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Split(1),),
            ),
        ),
        error=ExpectedError(match="input 1 .*incompatible.*Reshard"),
    ),
    OpCase(
        "partial_and_plain_slices",
        tensor.Stack(axis=0),
        (
            utils.make_shard_tensor_type(
                (8, 8),
                mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
                attrs=(types.Partial("sum"),),
            ),
            types.TensorType.from_shape((8, 8)),
        ),
        error=ExpectedError(match="input 1 does not carry Partial.*reshard"),
    ),
    OpCase(
        "axis_out_of_range",
        tensor.Stack(axis=-3),
        (types.TensorType.from_shape((8,)), types.TensorType.from_shape((8,))),
        error=ExpectedError(match="axis -3 out of range"),
    ),
)
test_op_case = op_case_test(CASES)
