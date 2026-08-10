"""Softplus evaluator value oracle + Partial(R) commutation typeinfer.

Two corpus functions apply ``softplus``, and the one that distinguishes it -- the
KDA forget gate -- sits behind a blocked Reference, so that comparison is not
running. The value oracle stays here until it is.
"""

from __future__ import annotations

import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Softplus()
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))


def test_softplus_rejects_partial_sum_input():
    """Softplus is monotone increasing: it commutes with max/min, not sum."""
    run_typeinfer_case(
        TypeInferCase("partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Softplus"))
    )


def test_softplus_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(
        EvalCase("softplus", Softplus(), (x,), torch.nn.functional.softplus(x), atol=1e-6)
    )
