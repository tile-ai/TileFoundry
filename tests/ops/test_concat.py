"""Concat's relation-driven sharded-layout boundary."""

from __future__ import annotations

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = make_mesh((4,))


def test_concat_propagates_a_split_outside_the_concat_axis():
    run_typeinfer_case(
        TypeInferCase(
            "non_concat_split_propagates",
            Concat(axis=0),
            (
                make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)),
                make_tensor_type((4, 8), _F),
            ),
            make_shard_tensor_type((8, 8), mesh=_M, attrs=(Split(1),)),
        )
    )


def test_concat_rejects_a_first_input_split_on_the_concat_axis():
    run_typeinfer_case(
        TypeInferCase(
            "first_input_concat_split_rejected",
            Concat(axis=0),
            (
                make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)),
                make_tensor_type((6, 8), _F),
            ),
            ExpectedError(match=r"input 0.*concat axis 0.*Reshard before Concat"),
        )
    )
