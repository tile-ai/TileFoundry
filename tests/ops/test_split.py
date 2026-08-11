"""Split produces equal, typed tuple fields with result-sized traffic."""

import math

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, infer_call, run_typeinfer_case
from tilefoundry.evaluator.context import EvalContext
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split as SplitAttr
from tilefoundry.ir.types.shard.shard_layout import shard_layout_local_shape
from tilefoundry.visitor_registry.contexts import TrafficBytes

CASES = [
    TypeInferCase(
        "negative_axis",
        Split(axis=-1, num_splits=2),
        (make_tensor_type((4, 8)),),
        TupleType(fields=(make_tensor_type((4, 4)),) * 2),
    ),
    TypeInferCase(
        "axis_out_of_range",
        Split(axis=2, num_splits=2),
        (make_tensor_type((4, 8)),),
        ExpectedError(match="axis 2 out of range"),
    ),
    TypeInferCase(
        "zero_splits",
        Split(axis=0, num_splits=0),
        (make_tensor_type((4, 8)),),
        ExpectedError(match="num_splits must be positive, got 0"),
    ),
    TypeInferCase(
        "negative_splits",
        Split(axis=0, num_splits=-2),
        (make_tensor_type((4, 8)),),
        ExpectedError(match="num_splits must be positive, got -2"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_split_typeinfer(case):
    run_typeinfer_case(case)


def test_split_rebuilds_plain_and_sharded_result_layouts():
    plain = make_tensor_type((16, 8), layout=Layout(shape=(16, 8), strides=(8, 1)))
    plain_parts = infer_call(Split(axis=0, num_splits=4), plain).fields
    assert all(part.layout == Layout(shape=(4, 8), strides=(8, 1)) for part in plain_parts)

    sharded = make_shard_tensor_type((16, 8), mesh=make_mesh((4,)), attrs=(SplitAttr(0),))
    sharded_parts = infer_call(Split(axis=1, num_splits=2), sharded).fields
    assert all(isinstance(part.layout, ShardLayout) for part in sharded_parts)
    assert all(part.layout.attrs == (SplitAttr(0),) for part in sharded_parts)
    assert all(
        math.prod(shard_layout_local_shape(part.layout)) == math.prod(part.shape) // 4
        for part in sharded_parts
    )


def test_split_evaluates_to_equal_tuple_fields() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    expected = tuple(torch.chunk(source, 3, dim=1))

    run_eval_case(EvalCase("three_columns", Split(axis=1, num_splits=3), (source,), expected))
    assert torch.equal(torch.cat(expected, dim=1), source)


def test_split_values_carry_their_exact_inferred_field_types() -> None:
    source_type = make_shard_tensor_type(
        (8, 4),
        dtype=DType.f32,
        storage="rmem",
        mesh=make_mesh((4,)),
        attrs=(SplitAttr(0),),
    )
    op = Split(axis=1, num_splits=2)
    result_type = infer_call(op, source_type)
    handler = eval_registry.lookup(Split)
    assert handler is not None

    result = handler(
        EvalContext(
            op=op,
            args=(TensorValue(torch.arange(32).reshape(8, 4).float(), source_type),),
            result_type=result_type,
        )
    )

    assert isinstance(result, TupleValue)
    assert tuple(element.type for element in result.elements) == result_type.fields
    assert not hasattr(result, "type")


def test_split_costs_one_full_read_and_write() -> None:
    moved = 4 * 6 * 4
    run_cost_case(
        CostCase(
            "split",
            Split(axis=1, num_splits=3),
            (make_tensor_type((4, 6)),),
            traffic=(TrafficBytes(read=moved), TrafficBytes(write=moved)),
        )
    )
