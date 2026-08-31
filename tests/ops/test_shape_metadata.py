"""HIR shape metadata reads concrete runtime extents without device traffic."""

from __future__ import annotations

import pytest
import torch

from tests.ops.cost_utils import CostCase, run_cost_case
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.tensor.rank import Rank
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.types import TensorType, make_tensor_type
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.contexts import TrafficBytes

_S = DimVar("runtime_shape", 1, 9)


@func
def _shape_metadata(x: Tensor[(_S, 4), "f32"]):
    return tf.shape_of(x), tf.rank(x)


def _idle(arity: int) -> tuple[TrafficBytes, ...]:
    return tuple(TrafficBytes() for _ in range(arity + 1))


COST_CASES = [
    CostCase("rank", Rank(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
    CostCase("shape_of", ShapeOf(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_shape_metadata_cost(case):
    run_cost_case(case)


def test_shape_metadata_uses_runtime_shape_and_host_types() -> None:
    actual_shape, actual_rank = evaluate(_shape_metadata, torch.zeros(3, 4))

    torch.testing.assert_close(actual_shape, torch.tensor([3, 4], dtype=torch.int64))
    torch.testing.assert_close(actual_rank, torch.tensor(2, dtype=torch.int64))
    calls = [expr for expr in collect_exprs(_shape_metadata.body) if isinstance(expr, Call)]
    metadata_calls = [
        call
        for call in calls
        if isinstance(call.target, (Rank, ShapeOf))
    ]
    assert metadata_calls
    assert all(call.type.storage is StorageKind.UMAT for call in metadata_calls)
    assert all(call.type.layout == EMPTY_LAYOUT for call in metadata_calls)


@func
def _shape_element_paths(x: Tensor[(_S, 4), "f32"]):
    return tf.shape_of(x)[1]


def test_shape_element_path_has_the_canonical_type() -> None:
    assert _shape_element_paths.body.type == TensorType.umat_scalar()
    actual = evaluate(_shape_element_paths, torch.zeros(3, 4))
    torch.testing.assert_close(actual, torch.tensor(4, dtype=torch.int64))
