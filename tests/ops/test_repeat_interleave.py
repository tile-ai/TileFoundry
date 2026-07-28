"""RepeatInterleave's fail-closed boundaries.

The growing axis invalidates the input layout, so a genuinely-sharded input
fails closed rather than emit a stale layout, and an out-of-range axis is named
rather than normalized.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = make_mesh((4,))

CASES = [
    # a genuine sharding cannot be re-expressed across the repeat -> fail closed.
    TypeInferCase(
        "sharded_fails_closed",
        RepeatInterleave(repeats=2, axis=1),
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)),),
        ExpectedError(match="cannot express a sharded layout"),
    ),
    TypeInferCase(
        "axis_out_of_range",
        RepeatInterleave(repeats=2, axis=5),
        (make_tensor_type((4,), _F),),
        ExpectedError(match="out of range"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_repeat_interleave_typeinfer(case):
    run_typeinfer_case(case)
