"""Split produces equal, typed tuple fields with result-sized traffic."""

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
from tilefoundry import core
from tilefoundry.interfaces import cost, evaluator
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor


def test_split_rebuilds_result_layouts():
    plain = types.TensorType.from_shape((16, 8), layout=types.Layout(shape=(16, 8), strides=(8, 1)))
    plain_parts = infer_call(tensor.Split(axis=0, num_splits=4), plain).fields
    assert all((part.layout == types.Layout(shape=(4, 8), strides=(8, 1)) for part in plain_parts))
    sharded = utils.make_shard_tensor_type(
        (16, 8), mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)), attrs=(types.Split(0),)
    )
    sharded_parts = infer_call(tensor.Split(axis=1, num_splits=2), sharded).fields
    assert all((isinstance(part.layout, types.ShardLayout) for part in sharded_parts))
    assert all((part.layout.attrs == (types.Split(0),) for part in sharded_parts))
    assert all(
        (
            math.prod(types.shard_layout_local_shape(part.layout)) == math.prod(part.shape) // 4
            for part in sharded_parts
        )
    )


def test_split_evaluates_equal_fields() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    expected = tuple(torch.chunk(source, 3, dim=1))
    check_op_case(OpCase("three_columns", tensor.Split(axis=1, num_splits=3), (source,), value=expected))
    assert torch.equal(torch.cat(expected, dim=1), source)


def test_split_values_carry_field_types() -> None:
    source_type = utils.make_shard_tensor_type(
        (8, 4),
        dtype=expr.DType.f32,
        storage="rmem",
        mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",)),
        attrs=(types.Split(0),),
    )
    op = tensor.Split(axis=1, num_splits=2)
    result_type = infer_call(op, source_type)
    handler = evaluator.registry.get(tensor.Split, None)
    assert handler is not None
    result = handler(
        evaluator.EvaluateContext(
            op=op,
            args=(evaluator.TensorValue(core.ShardTensor(source_type, torch.arange(32).reshape(8, 4).float())),),
            result_type=result_type,
        )
    )
    assert isinstance(result, evaluator.TupleValue)
    assert tuple((element.type for element in result.elements)) == result_type.fields
    assert result.type == result_type


def _make_split_costs_one_full_read_and_write() -> OpCase:
    moved = 4 * 6 * 4
    return OpCase(
        "split",
        tensor.Split(axis=1, num_splits=3),
        (types.TensorType.from_shape((4, 6)),),
        cost=Cost(traffic=(cost.TrafficBytes(read=moved), cost.TrafficBytes(write=moved))),
    )


CASES = (
    _make_split_costs_one_full_read_and_write(),
    OpCase(
        "negative_axis",
        tensor.Split(axis=-1, num_splits=2),
        (types.TensorType.from_shape((4, 8)),),
        type=types.TupleType(fields=(types.TensorType.from_shape((4, 4)),) * 2),
    ),
    OpCase(
        "axis_out_of_range",
        tensor.Split(axis=2, num_splits=2),
        (types.TensorType.from_shape((4, 8)),),
        error=ExpectedError(match="axis 2 out of range"),
    ),
    OpCase(
        "zero_splits",
        tensor.Split(axis=0, num_splits=0),
        (types.TensorType.from_shape((4, 8)),),
        error=ExpectedError(match="num_splits must be positive, got 0"),
    ),
    OpCase(
        "negative_splits",
        tensor.Split(axis=0, num_splits=-2),
        (types.TensorType.from_shape((4, 8)),),
        error=ExpectedError(match="num_splits must be positive, got -2"),
    ),
)
test_op_case = op_case_test(CASES)
