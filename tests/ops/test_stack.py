"""Stack derives fresh layouts through its input-to-slice relation."""

from __future__ import annotations

import math

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, infer_call, run_typeinfer_case
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Broadcast, Layout, Partial, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split, split_target_axes
from tilefoundry.visitor_registry.contexts import TrafficBytes


def test_stack_plain_result_has_a_fresh_layout_and_value() -> None:
    source_type = make_tensor_type(
        (2, 3), layout=Layout(shape=(2, 3), strides=(3, 1))
    )
    result = infer_call(Stack(axis=1), source_type, source_type)

    assert result.shape == (2, 2, 3)
    assert result.layout == Layout(shape=(2, 2, 3), strides=(6, 3, 1))
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = left + 10
    run_eval_case(
        EvalCase(
            "negative_axis",
            Stack(axis=-1),
            (left, right),
            torch.stack((left, right), dim=-1),
        )
    )


def test_stack_relation_carries_a_single_split_past_the_inserted_axis() -> None:
    mesh = make_mesh((4,))
    plain = make_tensor_type((2, 8))
    sharded = make_shard_tensor_type((2, 8), mesh=mesh, attrs=(Split(1),))

    result = infer_call(Stack(axis=0), plain, sharded)

    assert isinstance(result.layout, ShardLayout)
    assert result.layout != sharded.layout
    assert math.prod(result.layout.layout.shape) == math.prod(result.shape)
    assert split_target_axes(result.layout, result.shape) == (2,)


def test_stack_relation_carries_uniform_partial_slices() -> None:
    partial = make_shard_tensor_type(
        (2, 8), mesh=make_mesh((4,)), attrs=(Partial("sum"),)
    )

    result = infer_call(Stack(axis=1), partial, partial)

    assert isinstance(result.layout, ShardLayout)
    assert result.layout.layout.shape == result.shape
    assert result.layout.attrs == (Partial("sum"),)


CASES = [
    TypeInferCase(
        "incompatible_split_axes",
        Stack(axis=0),
        (
            make_shard_tensor_type((8, 8), mesh=make_mesh((4,)), attrs=(Split(0),)),
            make_shard_tensor_type((8, 8), mesh=make_mesh((4,)), attrs=(Split(1),)),
        ),
        ExpectedError(match=r"input 1 .*incompatible.*Reshard"),
    ),
    TypeInferCase(
        "incompatible_meshes",
        Stack(axis=0),
        (
            make_shard_tensor_type((8, 8), mesh=make_mesh((4,)), attrs=(Split(0),)),
            make_shard_tensor_type((8, 8), mesh=make_mesh((2,)), attrs=(Split(0),)),
        ),
        ExpectedError(match=r"input 1 references a different mesh.*Reshard"),
    ),
    TypeInferCase(
        "broadcast_operand_on_incompatible_mesh",
        Stack(axis=0),
        (
            make_shard_tensor_type(
                (8, 8), mesh=make_mesh((2,)), attrs=(Broadcast(),)
            ),
            make_shard_tensor_type(
                (8, 8), mesh=make_mesh((4,)), attrs=(Split(0),)
            ),
        ),
        ExpectedError(match=r"input 1 references a different mesh.*Reshard"),
    ),
    TypeInferCase(
        "partial_and_plain_slices",
        Stack(axis=0),
        (
            make_shard_tensor_type(
                (8, 8), mesh=make_mesh((4,)), attrs=(Partial("sum"),)
            ),
            make_tensor_type((8, 8)),
        ),
        ExpectedError(match=r"input 1 does not carry Partial.*Reshard"),
    ),
    TypeInferCase(
        "axis_out_of_range",
        Stack(axis=-3),
        (make_tensor_type((8,)), make_tensor_type((8,))),
        ExpectedError(match="axis -3 out of range"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_stack_rejects_underivable_inputs(case):
    run_typeinfer_case(case)


def test_stack_costs_each_input_read_and_the_result_write() -> None:
    input_bytes = 2 * 3 * 4
    run_cost_case(
        CostCase(
            "stack",
            Stack(axis=0),
            (make_tensor_type((2, 3), DType.f32),) * 2,
            traffic=(
                TrafficBytes(read=input_bytes),
                TrafficBytes(read=input_bytes),
                TrafficBytes(write=2 * input_bytes),
            ),
        )
    )
