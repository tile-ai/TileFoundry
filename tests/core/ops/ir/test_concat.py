"""Concat's relation-driven sharded-layout boundary."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Split

_F = DType.f32
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "non_concat_split_propagates",
        Concat(axis=0),
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)), make_tensor_type((4, 8), _F)),
        type=make_shard_tensor_type((8, 8), mesh=_M, attrs=(Split(1),)),
    ),
    OpCase(
        "first_input_concat_split_rejected",
        Concat(axis=0),
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)), make_tensor_type((6, 8), _F)),
        error=ExpectedError(match="input 0.*concat axis 0.*Reshard before Concat"),
    ),
)
test_op_case = op_case_test(CASES)
