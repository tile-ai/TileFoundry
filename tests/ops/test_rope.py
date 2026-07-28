"""RoPE's rejected shapes and its Partial semantics.

The rotated (q, k) value oracle lives in the model References: every corpus
decoder applies ``rope`` on its decode step, so a wrong rotation fails there.
What those models never build is an odd or mismatched head_dim, or a
``Partial``-carrying operand -- ``sum`` commutes with the rotation because it is
linear in q and k, ``max`` does not.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_BF = DType.bf16
_M = make_mesh((4,))


def _rope_inputs(q_shape, k_shape, *, q=None, k=None, cos=None, sin=None, pos=None):
    """The (q, k, cos, sin, pos) input types for a RoPE call."""
    return (
        q if q is not None else make_tensor_type(q_shape, _BF),
        k if k is not None else make_tensor_type(k_shape, _BF),
        cos if cos is not None else make_tensor_type((4096, q_shape[-1]), _BF),
        sin if sin is not None else make_tensor_type((4096, q_shape[-1]), _BF),
        pos if pos is not None else make_tensor_type((1,), DType.i32),
    )


CASES = [
    TypeInferCase(
        "odd_head_dim",
        RoPE(),
        _rope_inputs((1, 32, 127), (1, 4, 127)),
        ExpectedError(match="head_dim 127 must be even"),
    ),
    TypeInferCase(
        "mismatched_head_dims",
        RoPE(),
        _rope_inputs((1, 32, 128), (1, 4, 64)),
        ExpectedError(match="!= k head_dim"),
    ),
    # q and k are checked the same way, so q stands for both: a Partial(sum)
    # carries onto its own output field and leaves the other field plain, ...
    TypeInferCase(
        "partial_sum_q_passes",
        RoPE(),
        _rope_inputs(
            (1, 32, 128), (1, 4, 128),
            q=make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(Partial("sum"),), dtype=_BF),
        ),
        TupleType(
            fields=(
                make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(Partial("sum"),), dtype=_BF),
                make_tensor_type((1, 4, 128), _BF),
            )
        ),
    ),
    # ... and a Partial(max) does not commute with the rotation at all.
    TypeInferCase(
        "partial_max_q_errors",
        RoPE(),
        _rope_inputs(
            (1, 32, 128), (1, 4, 128),
            q=make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(Partial("max"),), dtype=_BF),
        ),
        ExpectedError(match="RoPE"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rope_typeinfer(case):
    run_typeinfer_case(case)
