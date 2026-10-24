"""Where's broadcast, placement-anchor, and causal-selection semantics."""

from __future__ import annotations

import pytest
import torch

from tests.core.evaluator.eval_utils import EvalCase, run_eval_case
from tests.core.ops.ir.cost_utils import CostCase, run_cost_case
from tests.core.ops.ir.typeinfer_utils import ExpectedError, TypeInferCase, run_typeinfer_case
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TrafficBytes

_MESH = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))


def test_where_evaluates_right_aligned_broadcast():
    condition = torch.tensor([True, False]).reshape(2, 1, 1)
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    other = torch.full((1, 3, 1), -1.0)

    run_eval_case(
        EvalCase(
            "broadcast",
            Where(),
            (condition, input_, other),
            torch.where(condition, input_, other),
        )
    )


TYPE_CASES = [
    TypeInferCase(
        "condition_does_not_anchor_storage",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage="gmem"),
            make_tensor_type((2, 8, 8), DType.f32, storage="rmem"),
            make_tensor_type((1, 8, 1), DType.f32, storage="rmem"),
        ),
        make_tensor_type(
            (2, 8, 8),
            DType.f32,
            layout=Layout((2, 8, 8), (64, 8, 1)),
            storage="rmem",
        ),
    ),
    TypeInferCase(
        "sharded_scores_anchor_distribution",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage=StorageKind.UMAT),
            make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
            make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
        ),
        make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
    ),
    TypeInferCase(
        "condition_must_be_bool",
        Where(),
        (
            make_tensor_type((2, 3), DType.i64),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        ExpectedError(match="condition must have bool dtype"),
    ),
    TypeInferCase(
        "branches_must_share_dtype",
        Where(),
        (
            make_tensor_type((2, 3), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f16),
        ),
        ExpectedError(match="data branch dtype mismatch"),
    ),
    TypeInferCase(
        "all_inputs_must_broadcast",
        Where(),
        (
            make_tensor_type((2, 4), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        ExpectedError(match="cannot broadcast"),
    ),
]


@pytest.mark.parametrize("case", TYPE_CASES, ids=lambda case: case.name)
def test_where_typeinfer(case):
    run_typeinfer_case(case)


def test_where_cost_counts_selection_and_materialization():
    """A select is a select, not a floating-point operation on its result.

    The result dtype is whatever was selected; the work is one selection per
    element, which the machine serves on its scalar path. Calling it a bool
    FLOP put work on the float pipe that never went there.
    """
    run_cost_case(
        CostCase(
            "where",
            Where(),
            (
                make_tensor_type((2, 3, 4), DType.bool),
                make_tensor_type((2, 3, 4), DType.f32),
                make_tensor_type((2, 3, 4), DType.f32),
            ),
            ops={"select": 24},
            traffic=(
                TrafficBytes(read=3),
                TrafficBytes(read=96),
                TrafficBytes(read=96),
                TrafficBytes(write=96),
            ),
        )
    )
