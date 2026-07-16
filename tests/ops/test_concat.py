"""Concat typeinfer: the concat axis sums; an all-unsharded concat produces an
unsharded output, and a genuinely-sharded input drops to an unsharded output
rather than carrying a fake layout onto the concatenated shape."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
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

CASES = [
    TypeInferCase(
        "unsharded",
        Concat(axis=0),
        (make_tensor_type((4, 8), _F), make_tensor_type((4, 8), _F)),
        make_tensor_type((8, 8), _F),
    ),
    # a genuine sharding on any input drops to None rather than carry a fake
    # layout onto the concatenated shape.
    TypeInferCase(
        "sharded_drops_layout",
        Concat(axis=0),
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)), make_tensor_type((4, 8), _F)),
        make_tensor_type((8, 8), _F),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_concat_typeinfer(case):
    run_typeinfer_case(case)


@pytest.mark.parametrize("axis", [0, 1], ids=["axis0", "axis1"])
def test_concat_evaluate(axis):
    torch.manual_seed(0)
    _ca, _cb = torch.randn(2, 3), torch.randn(2, 3)
    run_eval_case(EvalCase("", Concat(axis=axis), (_ca, _cb), torch.cat([_ca, _cb], dim=axis)))
