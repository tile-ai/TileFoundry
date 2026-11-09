"""Concat's relation-driven sharded-layout boundary."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor

_F = expr.DType.f32
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "non_concat_split_propagates",
        tensor.Concat(axis=0),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(1),)), types.TensorType.from_shape((4, 8), _F)),
        type=utils.make_shard_tensor_type((8, 8), mesh=_M, attrs=(types.Split(1),)),
    ),
    OpCase(
        "first_input_concat_split_rejected",
        tensor.Concat(axis=0),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(0),)), types.TensorType.from_shape((6, 8), _F)),
        error=ExpectedError(match="input 0.*concat axis 0.*Reshard before Concat"),
    ),
)
test_op_case = op_case_test(CASES)
