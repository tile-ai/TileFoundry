"""MoERoute typeinfer + Partial(R) commutation.

Routing is an opaque, data-dependent sort/permutation — no reduction
provably commutes, so a ``Partial`` on ``topk_ids`` is rejected regardless of
its reduction.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.nn.moe_route import MoERoute
from tilefoundry.ir.types import DType, TupleType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = MoERoute(num_experts=8, block_size=64)
_I = DType.i32

CASES = [
    TypeInferCase(
        "passthrough",
        _OP,
        (ten((256,), _I),),
        TupleType(
            fields=(
                ten((256,), _I),
                ten((9,), _I),
                ten((8,), _I),
            )
        ),
    ),
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (sharded((256,), (Partial("sum"),), mesh((4,)), dtype=_I),),
        ExpectedError(match="MoERoute", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_moe_route_typeinfer(case):
    run_typeinfer_case(case)
