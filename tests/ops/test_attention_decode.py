"""AttentionDecode typeinfer + Partial(R) commutation.

An opaque black box containing softmax internally — no reduction provably
commutes, so a ``Partial`` on ``q`` is rejected regardless of its reduction.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.nn.attention_decode import AttentionDecode
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = AttentionDecode()
_F = DType.f32
_Q = ten((1, 4, 128), _F)
_K = ten((1, 4, 128), _F)
_V = ten((1, 4, 128), _F)

CASES = [
    TypeInferCase("placeholder_mode_passthrough", _OP, (_Q, _K, _V), ten((1, 512), _F)),
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (sharded((1, 4, 128), (Partial("sum"),), mesh((4,))), _K, _V),
        ExpectedError(match="AttentionDecode", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_attention_decode_typeinfer(case):
    run_typeinfer_case(case)
