"""Slice's sharded-layout boundary: the sliced shape can no longer be described
by the input's layout, so a genuinely-sharded input drops to an unsharded output
rather than carrying a fake layout forward."""
from __future__ import annotations

from tests.ops.typeinfer_utils import (
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = make_mesh((4,))


def test_slice_of_sharded_input_drops_the_layout():
    run_typeinfer_case(
        TypeInferCase(
            "sharded_drops_layout",
            Slice(begin=(0, 0), end=(16, 16), strides=(1, 1)),
            (make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),)),),
            make_tensor_type((16, 16), _F),
        )
    )
