"""Binary's shard propagation and its Partial truth table.

Binary derives its output ``ShardLayout`` from the shared shard-propagation
engine: a layout mismatch between genuinely-sharded operands is an error, not a
silent lhs pick, and output storage anchors on concrete residency.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split

_ADD = Binary(kind=BinaryKind.ADD)
_MUL = Binary(kind=BinaryKind.MUL)
_SUB = Binary(kind=BinaryKind.SUB)
_F = DType.f32

# A single-axis mesh (g=4) for flat shards and a two-axis mesh (a=2, b=4) for
# factorized shards; cases reuse these so no test hand-builds a Mesh.
_M = make_mesh((4,))
_MAB = make_mesh((2, 4), ("a", "b"))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))
_BCAST = make_tensor_type((16, 8), _F)
_PSUM_AXIS0 = make_shard_tensor_type(
    (16, 8), mesh=_MAB, attrs=(Partial("sum"), Broadcast())
)
_PSUM_AXIS1 = make_shard_tensor_type(
    (16, 8), mesh=_MAB, attrs=(Broadcast(), Partial("sum"))
)

CASES = [
    # Right-aligned NumPy broadcast: a lower-rank operand against a higher-rank one.
    TypeInferCase("different_rank_broadcast", _ADD, (make_tensor_type((4, 8), _F), make_tensor_type((8,), _F)), make_tensor_type((4, 8), _F)),
    # lhs splits axis 0, rhs splits axis 1 on the same mesh axis → conflict,
    # not a silent lhs pick.
    TypeInferCase(
        "incompatible_split",
        _ADD,
        (
            make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)),
            make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(1),)),
        ),
        ExpectedError(match="incompatible"),
    ),
    # An unmaterialized literal operand abstains, but two *different* concrete
    # residencies have no anchor → error, not a pick.
    TypeInferCase(
        "conflicting_concrete_storage",
        _ADD,
        (make_tensor_type((4, 8), _F, storage="gmem"), make_tensor_type((4, 8), _F, storage="rmem")),
        ExpectedError(match="conflicting storage"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_binary_typeinfer(case):
    run_typeinfer_case(case)


PARTIAL_CASES = [
    TypeInferCase("add_partial_sum_partial_sum_passes", _ADD, (_PSUM, _PSUM), _PSUM),
    TypeInferCase(
        "add_partial_max_partial_max_errors",
        _ADD,
        (_PMAX, _PMAX),
        ExpectedError(match="do not commute"),
    ),
    TypeInferCase(
        "add_partial_sum_broadcast_errors",
        _ADD,
        (_PSUM, _BCAST),
        ExpectedError(match="carries Partial"),
    ),
    TypeInferCase("add_partial_max_broadcast_passes", _ADD, (_PMAX, _BCAST), _PMAX),
    TypeInferCase("mul_partial_sum_broadcast_passes", _MUL, (_PSUM, _BCAST), _PSUM),
    TypeInferCase(
        "mul_partial_max_broadcast_errors",
        _MUL,
        (_PMAX, _BCAST),
        ExpectedError(match="carries Partial"),
    ),
    TypeInferCase(
        "sub_partial_sum_broadcast_errors",
        _SUB,
        (_PSUM, _BCAST),
        ExpectedError(match="carries Partial"),
    ),
    TypeInferCase(
        "partial_sum_different_mesh_axes_errors",
        _ADD,
        (_PSUM_AXIS0, _PSUM_AXIS1),
        ExpectedError(match="mesh axis 0"),
    ),
]


@pytest.mark.parametrize("case", PARTIAL_CASES, ids=lambda c: c.name)
def test_binary_partial_typeinfer(case):
    run_typeinfer_case(case)


def test_lower_rank_split_right_aligns():
    """A lower-rank sharded operand's Split lands on the *output* axis it
    right-aligns to, whichever side carries it. Checked as which mesh axis holds
    Split on that output axis, not as the internal layout position count a valid
    derivation happens to produce."""
    split_1d = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
    plain_2d = make_tensor_type((4, 8), _F)
    for lhs, rhs in ((plain_2d, split_1d), (split_1d, plain_2d)):
        out = infer_call(_ADD, lhs, rhs)
        assert out.shape == (4, 8)
        assert out.layout.attrs == (Split(1),)
