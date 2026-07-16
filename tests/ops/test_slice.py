"""Slice typeinfer: shape shrinks per the begin/end/strides; an unsharded
input slices to an unsharded output, and a genuinely-sharded input drops to an
unsharded output rather than carrying a fake layout onto the sliced shape."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
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

CASES = [
    TypeInferCase(
        "unsharded",
        Slice(begin=(0, 0), end=(4, 8), strides=(1, 1)),
        (make_tensor_type((4, 16), _F),),
        make_tensor_type((4, 8), _F),
    ),
    # a genuine sharding drops to None rather than carry a fake layout.
    TypeInferCase(
        "sharded_drops_layout",
        Slice(begin=(0, 0), end=(16, 16), strides=(1, 1)),
        (make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),)),),
        make_tensor_type((16, 16), _F),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_slice_typeinfer(case):
    run_typeinfer_case(case)


def test_slice_evaluate():
    esl = torch.arange(8.0)
    run_eval_case(
        EvalCase("strided", Slice(begin=(1,), end=(7,), strides=(2,)), (esl,), esl[1:7:2])
    )
