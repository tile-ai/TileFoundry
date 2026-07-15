"""RepeatInterleave typeinfer: the named axis grows by ``repeats``.

The growing axis invalidates the input cute layout, so a sharded layout is not
carried: an unsharded or fully-replicated input yields an unsharded output, and
a genuinely-sharded input fails closed rather than emit a stale layout.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Split

_F = DType.f32
_M = mesh((4,))

CASES = [
    TypeInferCase(
        "unsharded_grows_axis",
        RepeatInterleave(repeats=2, axis=1),
        (ten((4, 8), _F),),
        ten((4, 16), _F),
    ),
    # a fully-replicated input is logically plain -> unsharded output.
    TypeInferCase(
        "replicated_drops_to_none",
        RepeatInterleave(repeats=2, axis=1),
        (sharded((4, 8), (Broadcast(),), _M),),
        ten((4, 16), _F),
    ),
    # a genuine sharding cannot be re-expressed across the repeat -> fail closed.
    TypeInferCase(
        "sharded_fails_closed",
        RepeatInterleave(repeats=2, axis=1),
        (sharded((4, 8), (Split(0),), _M),),
        ExpectedError(match="cannot express a sharded layout"),
    ),
    TypeInferCase(
        "axis_out_of_range",
        RepeatInterleave(repeats=2, axis=5),
        (ten((4,), _F),),
        ExpectedError(match="out of range"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_repeat_interleave_typeinfer(case):
    run_typeinfer_case(case)


@pytest.mark.parametrize("op", [RepeatInterleave(repeats=2, axis=0)], ids=["rep2"])
def test_repeat_interleave_evaluate(op):
    x = torch.tensor([1.0, 2.0, 3.0])
    run_eval_case(EvalCase("", op, (x,), torch.repeat_interleave(x, 2, dim=0)))
