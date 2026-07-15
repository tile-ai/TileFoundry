"""MoEExpertCompute typeinfer + Partial(R) commutation.

An opaque black box (FP8 quant, fused GEMMs, SiLU, weighted expert sum) —
none of its internals are proven to commute with any reduction, so a
``Partial`` on ``x`` is rejected regardless of its reduction.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.nn.moe_expert_compute import MoEExpertCompute
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = MoEExpertCompute(num_experts=8, topk=2, intermediate=512)
_F = DType.f32
_I = DType.i32
_X = ten((4, 256), _F)
_REST = (
    ten((4, 2), _I),  # topk_ids
    ten((4, 2), _F),  # topk_weights
    ten((4, 8), _F),  # routing
    ten((8, 512, 256), _F),  # w_gate
    ten((8, 4, 256), _F),  # w_gate_scale
    ten((8, 512, 256), _F),  # w_up
    ten((8, 4, 256), _F),  # w_up_scale
    ten((8, 256, 512), _F),  # w_down
    ten((8, 2, 512), _F),  # w_down_scale
)

CASES = [
    TypeInferCase("passthrough", _OP, (_X, *_REST), _X),
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (sharded((4, 256), (Partial("sum"),), mesh((4,))), *_REST),
        ExpectedError(match="MoEExpertCompute", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_moe_expert_compute_typeinfer(case):
    run_typeinfer_case(case)
