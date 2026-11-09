"""RoPE's rejected shapes and its Partial semantics.

The rotated (q, k) value oracle lives in the model References: every corpus
decoder applies ``rope`` on its decode step, so a wrong rotation fails there.
What those models never build is an odd or mismatched head_dim, a q and k of
different ranks, or a ``Partial``-carrying operand -- ``sum`` commutes with the
rotation because it is linear in q and k, ``max`` does not.
"""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import nn

_BF = expr.DType.bf16
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))


def _rope_inputs(q_shape, k_shape, *, q=None, k=None, cos=None, sin=None, pos=None):
    """The (q, k, cos, sin, pos) input types for a RoPE call."""
    return (
        q if q is not None else types.TensorType.from_shape(q_shape, _BF),
        k if k is not None else types.TensorType.from_shape(k_shape, _BF),
        cos if cos is not None else types.TensorType.from_shape((4096, q_shape[-1]), _BF),
        sin if sin is not None else types.TensorType.from_shape((4096, q_shape[-1]), _BF),
        pos if pos is not None else types.TensorType.from_shape((1,), expr.DType.i32),
    )


CASES = (
    OpCase(
        "odd_head_dim",
        nn.RoPE(),
        _rope_inputs((1, 32, 127), (1, 4, 127)),
        error=ExpectedError(match="head_dim 127 must be even"),
    ),
    OpCase(
        "mismatched_head_dims",
        nn.RoPE(),
        _rope_inputs((1, 32, 128), (1, 4, 64)),
        error=ExpectedError(match="!= k head_dim"),
    ),
    OpCase(
        "mismatched_ranks",
        nn.RoPE(),
        _rope_inputs((1, 32, 128), (32, 128)),
        error=ExpectedError(match="q rank 3 != k rank 2"),
    ),
    OpCase(
        "partial_sum_q_passes",
        nn.RoPE(),
        _rope_inputs(
            (1, 32, 128),
            (1, 4, 128),
            q=utils.make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(types.Partial("sum"),), dtype=_BF),
        ),
        type=types.TupleType(
            fields=(
                utils.make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(types.Partial("sum"),), dtype=_BF),
                types.TensorType.from_shape((1, 4, 128), _BF),
            )
        ),
    ),
    OpCase(
        "partial_max_q_errors",
        nn.RoPE(),
        _rope_inputs(
            (1, 32, 128),
            (1, 4, 128),
            q=utils.make_shard_tensor_type((1, 32, 128), mesh=_M, attrs=(types.Partial("max"),), dtype=_BF),
        ),
        error=ExpectedError(match="RoPE"),
    ),
)
test_op_case = op_case_test(CASES)
