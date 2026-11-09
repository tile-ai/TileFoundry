"""Where's broadcast, placement-anchor, and causal-selection semantics."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, op_case_test
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor

_MESH = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))


def _make_where_evaluates_right_aligned_broadcast():
    condition = torch.tensor([True, False]).reshape(2, 1, 1)
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    other = torch.full((1, 3, 1), -1.0)
    return OpCase(
        "broadcast",
        tensor.Where(),
        (condition, input_, other),
        value=torch.where(condition, input_, other),
    )


CASES = (
    _make_where_evaluates_right_aligned_broadcast(),
    OpCase(
        "where",
        tensor.Where(),
        (
            types.TensorType.from_shape((2, 3, 4), expr.DType.bool),
            types.TensorType.from_shape((2, 3, 4), expr.DType.f32),
            types.TensorType.from_shape((2, 3, 4), expr.DType.f32),
        ),
        cost=Cost(
            ops={"select": 24},
            traffic=(
                cost.TrafficBytes(read=3),
                cost.TrafficBytes(read=96),
                cost.TrafficBytes(read=96),
                cost.TrafficBytes(write=96),
            ),
        ),
    ),
    OpCase(
        "condition_does_not_anchor_storage",
        tensor.Where(),
        (
            types.TensorType.from_shape((8, 8), expr.DType.bool, storage="gmem"),
            types.TensorType.from_shape((2, 8, 8), expr.DType.f32, storage="rmem"),
            types.TensorType.from_shape((1, 8, 1), expr.DType.f32, storage="rmem"),
        ),
        type=types.TensorType.from_shape(
            (2, 8, 8), expr.DType.f32, layout=types.Layout((2, 8, 8), (64, 8, 1)), storage="rmem"
        ),
    ),
    OpCase(
        "sharded_scores_anchor_distribution",
        tensor.Where(),
        (
            types.TensorType.from_shape((8, 8), expr.DType.bool, storage=expr.StorageKind.UMAT),
            utils.make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(types.Split(1),), storage="smem"),
            utils.make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(types.Split(1),), storage="smem"),
        ),
        type=utils.make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(types.Split(1),), storage="smem"),
    ),
    OpCase(
        "condition_must_be_bool",
        tensor.Where(),
        (
            types.TensorType.from_shape((2, 3), expr.DType.i64),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
        ),
        error=ExpectedError(match="condition must have bool dtype"),
    ),
    OpCase(
        "branches_must_share_dtype",
        tensor.Where(),
        (
            types.TensorType.from_shape((2, 3), expr.DType.bool),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
            types.TensorType.from_shape((2, 3), expr.DType.f16),
        ),
        error=ExpectedError(match="data branch dtype mismatch"),
    ),
    OpCase(
        "all_inputs_must_broadcast",
        tensor.Where(),
        (
            types.TensorType.from_shape((2, 4), expr.DType.bool),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
        ),
        error=ExpectedError(match="cannot broadcast"),
    ),
)
test_op_case = op_case_test(CASES)
