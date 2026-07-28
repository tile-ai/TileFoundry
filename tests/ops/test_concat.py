"""Concat's sharded-layout boundary: a genuine sharding on any input drops to an
unsharded output rather than carrying a fake layout onto the concatenated shape."""
from __future__ import annotations

from tests.ops.typeinfer_utils import (
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = make_mesh((4,))


def test_concat_of_a_sharded_input_drops_the_layout():
    run_typeinfer_case(
        TypeInferCase(
            "sharded_drops_layout",
            Concat(axis=0),
            (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)), make_tensor_type((4, 8), _F)),
            make_tensor_type((8, 8), _F),
        )
    )
