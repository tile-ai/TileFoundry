"""Slice typeinfer: shape shrinks per the begin/end/strides; an unsharded
input slices to an unsharded output, and a genuinely-sharded input drops to an
unsharded output rather than carrying a fake layout onto the sliced shape."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = mesh((4,))

CASES = [
    TypeInferCase(
        "unsharded",
        Slice(begin=(0, 0), end=(4, 8), strides=(1, 1)),
        (ten((4, 16), _F),),
        ten((4, 8), _F),
    ),
    # a genuine sharding drops to None rather than carry a fake layout.
    TypeInferCase(
        "sharded_drops_layout",
        Slice(begin=(0, 0), end=(16, 16), strides=(1, 1)),
        (sharded((16, 32), (Split(0),), _M),),
        ten((16, 16), _F),
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
