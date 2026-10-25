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
from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    ShardLayout,
    Topology,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Split as SplitAttr
from tilefoundry.ir.types.shard_layout import shard_layout_local_shape
from tilefoundry.visitor_registry.contexts import TrafficBytes


def test_split_rebuilds_result_layouts():
    plain = make_tensor_type((16, 8), layout=Layout(shape=(16, 8), strides=(8, 1)))
    plain_parts = infer_call(Split(axis=0, num_splits=4), plain).fields
    assert all((part.layout == Layout(shape=(4, 8), strides=(8, 1)) for part in plain_parts))
    sharded = make_shard_tensor_type(
        (16, 8), mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)), attrs=(SplitAttr(0),)
    )
    sharded_parts = infer_call(Split(axis=1, num_splits=2), sharded).fields
    assert all((isinstance(part.layout, ShardLayout) for part in sharded_parts))
    assert all((part.layout.attrs == (SplitAttr(0),) for part in sharded_parts))
    assert all(
        (
            math.prod(shard_layout_local_shape(part.layout)) == math.prod(part.shape) // 4
            for part in sharded_parts
        )
    )


def test_split_evaluates_equal_fields() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    expected = tuple(torch.chunk(source, 3, dim=1))
    check_op_case(OpCase("three_columns", Split(axis=1, num_splits=3), (source,), value=expected))
    assert torch.equal(torch.cat(expected, dim=1), source)


def test_split_values_carry_field_types() -> None:
    source_type = make_shard_tensor_type(
        (8, 4),
        dtype=DType.f32,
        storage="rmem",
        mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
        attrs=(SplitAttr(0),),
    )
    op = Split(axis=1, num_splits=2)
    result_type = infer_call(op, source_type)
    handler = eval_registry.lookup(Split)
    assert handler is not None
    result = handler(
        EvaluateContext(
            op=op,
            args=(TensorValue(torch.arange(32).reshape(8, 4).float(), source_type),),
            result_type=result_type,
        )
    )
    assert isinstance(result, TupleValue)
    assert tuple((element.type for element in result.elements)) == result_type.fields
    assert not hasattr(result, "type")


def _make_split_costs_one_full_read_and_write() -> OpCase:
    moved = 4 * 6 * 4
    return OpCase(
        "split",
        Split(axis=1, num_splits=3),
        (make_tensor_type((4, 6)),),
        cost=Cost(traffic=(TrafficBytes(read=moved), TrafficBytes(write=moved))),
    )


CASES = (
    _make_split_costs_one_full_read_and_write(),
    OpCase(
        "negative_axis",
        Split(axis=-1, num_splits=2),
        (make_tensor_type((4, 8)),),
        type=TupleType(fields=(make_tensor_type((4, 4)),) * 2),
    ),
    OpCase(
        "axis_out_of_range",
        Split(axis=2, num_splits=2),
        (make_tensor_type((4, 8)),),
        error=ExpectedError(match="axis 2 out of range"),
    ),
    OpCase(
        "zero_splits",
        Split(axis=0, num_splits=0),
        (make_tensor_type((4, 8)),),
        error=ExpectedError(match="num_splits must be positive, got 0"),
    ),
    OpCase(
        "negative_splits",
        Split(axis=0, num_splits=-2),
        (make_tensor_type((4, 8)),),
        error=ExpectedError(match="num_splits must be positive, got -2"),
    ),
)
test_op_case = op_case_test(CASES)
