"""Where's broadcast, placement-anchor, and causal-selection semantics."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, op_case_test
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


def _make_where_evaluates_right_aligned_broadcast():
    condition = torch.tensor([True, False]).reshape(2, 1, 1)
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    other = torch.full((1, 3, 1), -1.0)
    return OpCase(
        "broadcast",
        Where(),
        (condition, input_, other),
        value=torch.where(condition, input_, other),
    )


CASES = (
    _make_where_evaluates_right_aligned_broadcast(),
    OpCase(
        "where",
        Where(),
        (
            make_tensor_type((2, 3, 4), DType.bool),
            make_tensor_type((2, 3, 4), DType.f32),
            make_tensor_type((2, 3, 4), DType.f32),
        ),
        cost=Cost(
            ops={"select": 24},
            traffic=(
                TrafficBytes(read=3),
                TrafficBytes(read=96),
                TrafficBytes(read=96),
                TrafficBytes(write=96),
            ),
        ),
    ),
    OpCase(
        "condition_does_not_anchor_storage",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage="gmem"),
            make_tensor_type((2, 8, 8), DType.f32, storage="rmem"),
            make_tensor_type((1, 8, 1), DType.f32, storage="rmem"),
        ),
        type=make_tensor_type(
            (2, 8, 8), DType.f32, layout=Layout((2, 8, 8), (64, 8, 1)), storage="rmem"
        ),
    ),
    OpCase(
        "sharded_scores_anchor_distribution",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage=StorageKind.UMAT),
            make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
            make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
        ),
        type=make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"),
    ),
    OpCase(
        "condition_must_be_bool",
        Where(),
        (
            make_tensor_type((2, 3), DType.i64),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        error=ExpectedError(match="condition must have bool dtype"),
    ),
    OpCase(
        "branches_must_share_dtype",
        Where(),
        (
            make_tensor_type((2, 3), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f16),
        ),
        error=ExpectedError(match="data branch dtype mismatch"),
    ),
    OpCase(
        "all_inputs_must_broadcast",
        Where(),
        (
            make_tensor_type((2, 4), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        error=ExpectedError(match="cannot broadcast"),
    ),
)
test_op_case = op_case_test(CASES)
