"""HIR shape metadata reads concrete runtime extents without device traffic."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tilefoundry import func
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.shape.shape_compose import ShapeCompose
from tilefoundry.ir.hir.shape.shape_extract import ShapeExtract
from tilefoundry.ir.hir.tensor.rank import Rank
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.types import DType, TensorType, make_tensor_type
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.visitor_registry.contexts import TrafficBytes

_S = DimVar("runtime_shape", 1, 9)


@func
def _shape_metadata(x: Tensor[(_S, 4), "f32"]):
    shape = tf.shape_of(x)
    leading = tf.shape_extract(shape, index=0)
    trailing = tf.shape_extract(shape, index=1)
    return tf.shape_compose(leading, trailing, tf.rank(x))


def _idle(arity: int) -> tuple[TrafficBytes, ...]:
    return tuple(TrafficBytes() for _ in range(arity + 1))


_SHAPE_TYPE = TensorType(
    shape=(2,), dtype=DType.i64, layout=EMPTY_LAYOUT, storage=None
)


COST_CASES = [
    CostCase("rank", Rank(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
    CostCase("shape_of", ShapeOf(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
    CostCase("shape_compose_empty", ShapeCompose(), (), traffic=_idle(0)),
    CostCase("shape_extract", ShapeExtract(index=0), (_SHAPE_TYPE,), traffic=_idle(1)),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_shape_metadata_cost(case):
    run_cost_case(case)


def test_shape_metadata_uses_runtime_shape_and_host_types() -> None:
    actual = evaluate(_shape_metadata, torch.zeros(3, 4), device="cpu")

    torch.testing.assert_close(actual, torch.tensor([3, 4, 2], dtype=torch.int64))
    calls = [expr for expr in postorder(_shape_metadata.body) if isinstance(expr, Call)]
    metadata_calls = [
        call
        for call in calls
        if isinstance(call.target, (Rank, ShapeOf, ShapeCompose, ShapeExtract))
    ]
    assert metadata_calls
    assert all(call.type.storage is None for call in metadata_calls)
    assert all(call.type.layout == EMPTY_LAYOUT for call in metadata_calls)


def test_shape_compose_preserves_the_empty_shape() -> None:
    run_eval_case(
        EvalCase(
            "empty_shape",
            ShapeCompose(),
            (),
            torch.empty((0,), dtype=torch.int64),
        )
    )
