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
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.ir.types.storage import StorageKind

_ADD = Binary(kind=BinaryKind.ADD)
_MUL = Binary(kind=BinaryKind.MUL)
_SUB = Binary(kind=BinaryKind.SUB)
_F = DType.f32


_M = make_mesh((4,))
_MAB = make_mesh((2, 4), ("a", "b"))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))
_BCAST = make_tensor_type((16, 8), _F)
_PSUM_AXIS0 = make_shard_tensor_type((16, 8), mesh=_MAB, attrs=(Partial("sum"), Broadcast()))
_PSUM_AXIS1 = make_shard_tensor_type((16, 8), mesh=_MAB, attrs=(Broadcast(), Partial("sum")))

CASES = [
    TypeInferCase(
        "different_rank_broadcast",
        _ADD,
        (make_tensor_type((4, 8), _F), make_tensor_type((8,), _F)),
        make_tensor_type((4, 8), _F),
    ),
    TypeInferCase(
        "empty_axis",
        _ADD,
        (make_tensor_type((1, 0, 8), _F), make_tensor_type((8,), _F)),
        make_tensor_type((1, 0, 8), _F),
    ),
    TypeInferCase(
        "conflicting_concrete_storage",
        _ADD,
        (
            make_tensor_type((4, 8), _F, storage="gmem"),
            make_tensor_type((4, 8), _F, storage="rmem"),
        ),
        ExpectedError(match="conflicting storage"),
    ),
    TypeInferCase(
        "unmaterialized_abstains_from_storage",
        _ADD,
        (
            make_tensor_type((4, 8), _F, storage=StorageKind.UMAT),
            make_tensor_type((4, 8), _F, storage="rmem"),
        ),
        make_tensor_type(
            (4, 8), _F, layout=Layout((4, 8), (8, 1)), storage="rmem"
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_binary_typeinfer(case):
    run_typeinfer_case(case)


def test_broadcast_layout_fallback_does_not_weaken_shard_mismatch():
    narrow = (1, 1, 16, 1, 128)
    wide = (1, 1, 16, 4, 128)
    lhs = make_tensor_type(narrow, _F, layout=Layout(narrow))
    rhs = make_tensor_type(wide, _F, layout=Layout(wide))
    assert infer_call(_MUL, lhs, rhs).layout is None
    assert infer_call(_MUL, lhs, make_tensor_type(wide, _F)).layout is None

    replicated = make_shard_tensor_type(narrow, mesh=_M, attrs=(Broadcast(),))
    replicated_out = infer_call(_MUL, replicated, rhs)
    assert replicated_out.layout.layout.shape == wide

    split_0 = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),))
    split_1 = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(1),))
    with pytest.raises(VerifyError, match="incompatible"):
        infer_call(_ADD, split_0, split_1)


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
    """Test lower rank split right aligns.

    A lower-rank sharded operand's Split lands on the *output* axis it
    right-aligns to, whichever side carries it. Checked as which mesh axis holds
    Split on that output axis, not as the internal layout position count a valid
    derivation happens to produce.
    """
    split_1d = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
    plain_2d = make_tensor_type((4, 8), _F)
    for lhs, rhs in ((plain_2d, split_1d), (split_1d, plain_2d)):
        out = infer_call(_ADD, lhs, rhs)
        assert out.shape == (4, 8)
        assert out.layout.attrs == (Split(1),)
