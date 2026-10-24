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
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Split, split_target_axes
from tilefoundry.visitor_registry.contexts import TrafficBytes


def test_stack_plain_result_has_a_fresh_layout_and_value() -> None:
    source_type = make_tensor_type((2, 3), layout=Layout(shape=(2, 3), strides=(3, 1)))
    result = infer_call(Stack(axis=1), source_type, source_type)
    assert result.shape == (2, 2, 3)
    assert result.layout == Layout(shape=(2, 2, 3), strides=(6, 3, 1))
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = left + 10
    check_op_case(
        OpCase(
            "negative_axis", Stack(axis=-1), (left, right), value=torch.stack((left, right), dim=-1)
        )
    )


def test_stack_relation_carries_a_single_split_past_the_inserted_axis() -> None:
    mesh = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
    plain = make_tensor_type((2, 8))
    sharded = make_shard_tensor_type((2, 8), mesh=mesh, attrs=(Split(1),))
    result = infer_call(Stack(axis=0), plain, sharded)
    assert isinstance(result.layout, ShardLayout)
    assert result.layout != sharded.layout
    assert math.prod(result.layout.layout.shape) == math.prod(result.shape)
    assert split_target_axes(result.layout, result.shape) == (2,)


def test_stack_relation_carries_uniform_partial_slices() -> None:
    partial = make_shard_tensor_type(
        (2, 8),
        mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
        attrs=(Partial("sum"),),
    )
    result = infer_call(Stack(axis=1), partial, partial)
    assert isinstance(result.layout, ShardLayout)
    assert result.layout.layout.shape == result.shape
    assert result.layout.attrs == (Partial("sum"),)


def _make_stack_costs_each_input_read_and_the_result_write() -> OpCase:
    input_bytes = 2 * 3 * 4
    return OpCase(
        "stack",
        Stack(axis=0),
        (make_tensor_type((2, 3), DType.f32),) * 2,
        cost=Cost(
            traffic=(
                TrafficBytes(read=input_bytes),
                TrafficBytes(read=input_bytes),
                TrafficBytes(write=2 * input_bytes),
            )
        ),
    )


CASES = (
    _make_stack_costs_each_input_read_and_the_result_write(),
    OpCase(
        "incompatible_split_axes",
        Stack(axis=0),
        (
            make_shard_tensor_type(
                (8, 8),
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Split(0),),
            ),
            make_shard_tensor_type(
                (8, 8),
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Split(1),),
            ),
        ),
        error=ExpectedError(match="input 1 .*incompatible.*Reshard"),
    ),
    OpCase(
        "partial_and_plain_slices",
        Stack(axis=0),
        (
            make_shard_tensor_type(
                (8, 8),
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Partial("sum"),),
            ),
            make_tensor_type((8, 8)),
        ),
        error=ExpectedError(match="input 1 does not carry Partial.*Reshard"),
    ),
    OpCase(
        "axis_out_of_range",
        Stack(axis=-3),
        (make_tensor_type((8,)), make_tensor_type((8,))),
        error=ExpectedError(match="axis -3 out of range"),
    ),
)
test_op_case = op_case_test(CASES)
