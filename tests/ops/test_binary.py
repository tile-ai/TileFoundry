"""Binary typeinfer over the relation-driven path.

Binary derives its output shape by right-aligned NumPy broadcast and its
output ``ShardLayout`` from the shared shard-propagation engine. A layout
mismatch between genuinely-sharded operands is an error (no silent lhs pick);
replicated operands and unsharded layouts pass through.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimVar
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
    # ── shape inference (unsharded) ──────────────────────────────────────────
    TypeInferCase("same_shape", _ADD, (make_tensor_type((4, 8), _F), make_tensor_type((4, 8), _F)), make_tensor_type((4, 8), _F)),
    TypeInferCase("size1_broadcast", _ADD, (make_tensor_type((4, 8), _F), make_tensor_type((1, 8), _F)), make_tensor_type((4, 8), _F)),
    TypeInferCase("different_rank_broadcast", _ADD, (make_tensor_type((4, 8), _F), make_tensor_type((8,), _F)), make_tensor_type((4, 8), _F)),
    TypeInferCase("scalar_broadcast", _ADD, (make_tensor_type((), _F), make_tensor_type((4, 8), _F)), make_tensor_type((4, 8), _F)),
    TypeInferCase(
        "dynamic_dim",
        _ADD,
        (make_tensor_type((DimVar("N", 1, 64), 8), _F), make_tensor_type((DimVar("N", 1, 64), 8), _F)),
        make_tensor_type((DimVar("N", 1, 64), 8), _F),
    ),
    TypeInferCase(
        "dtype_mismatch",
        _ADD,
        (make_tensor_type((4, 8), _F), make_tensor_type((4, 8), DType.bf16)),
        ExpectedError(match="dtype mismatch"),
    ),
    # ── shard propagation ────────────────────────────────────────────────────
    # lhs split, rhs replicated → output keeps lhs's split.
    TypeInferCase(
        "sharded_lhs_replicated_rhs",
        _ADD,
        (make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)), make_tensor_type((16, 8), _F)),
        make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)),
    ),
    # both split the same axis identically → that split.
    TypeInferCase(
        "both_split_same_axis",
        _ADD,
        (
            make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)),
            make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)),
        ),
        make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)),
    ),
    # split side + broadcast side: lhs (4,8) split axis 0, rhs (8,) broadcasts.
    TypeInferCase(
        "split_side_plus_broadcast_side",
        _ADD,
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)), make_tensor_type((8,), _F)),
        make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)),
    ),
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
    # two mesh axes split the same tensor axis (neither operand supplies both):
    # the output factorizes axis 0 into one sub-position per mesh extent.
    TypeInferCase(
        "two_mesh_axes_synthesize_factorized",
        _ADD,
        (
            make_shard_tensor_type((8,), mesh=_MAB, attrs=(Split(0), Broadcast())),
            make_shard_tensor_type((8,), mesh=_MAB, attrs=(Broadcast(), Split(0))),
        ),
        make_shard_tensor_type((8,), mesh=_MAB, attrs=(Split(0), Split(0))),
    ),
    # one operand already carries the full factorized layout → carried through.
    TypeInferCase(
        "factorized_input_passes_through",
        _ADD,
        (
            make_shard_tensor_type((8,), mesh=_MAB, attrs=(Split(0), Split(0))),
            make_tensor_type((8,), _F),
        ),
        make_shard_tensor_type((8,), mesh=_MAB, attrs=(Split(0), Split(0))),
    ),
    # ── output storage (anchor on concrete residency) ────────────────────────
    # An unmaterialized literal operand (storage=umat) abstains; the concrete
    # operand anchors the output, independent of operand order.
    TypeInferCase(
        "literal_rhs_anchors_gmem",
        _ADD,
        (make_tensor_type((4, 8), _F, storage="gmem"), make_tensor_type((), _F, storage=StorageKind.UMAT)),
        make_tensor_type((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "literal_lhs_anchors_gmem",
        _ADD,
        (make_tensor_type((), _F, storage=StorageKind.UMAT), make_tensor_type((4, 8), _F, storage="gmem")),
        make_tensor_type((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "both_gmem",
        _ADD,
        (make_tensor_type((4, 8), _F, storage="gmem"), make_tensor_type((4, 8), _F, storage="gmem")),
        make_tensor_type((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "both_rmem",
        _ADD,
        (make_tensor_type((4, 8), _F, storage="rmem"), make_tensor_type((4, 8), _F, storage="rmem")),
        make_tensor_type((4, 8), _F, storage="rmem"),
    ),
    # All operands unmaterialized (e.g. `1 + 1`) → output stays unmaterialized.
    TypeInferCase(
        "all_unmaterialized",
        _ADD,
        (make_tensor_type((), _F, storage=StorageKind.UMAT), make_tensor_type((), _F, storage=StorageKind.UMAT)),
        make_tensor_type((), _F, storage=StorageKind.UMAT),
    ),
    # Two different concrete residencies have no anchor → error, not a pick.
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
        ExpectedError(match="Binary ADD"),
    ),
    TypeInferCase(
        "add_partial_sum_broadcast_errors",
        _ADD,
        (_PSUM, _BCAST),
        ExpectedError(match="Binary ADD"),
    ),
    TypeInferCase("add_partial_max_broadcast_passes", _ADD, (_PMAX, _BCAST), _PMAX),
    TypeInferCase("mul_partial_sum_broadcast_passes", _MUL, (_PSUM, _BCAST), _PSUM),
    TypeInferCase(
        "mul_partial_max_broadcast_errors",
        _MUL,
        (_PMAX, _BCAST),
        ExpectedError(match="Binary MUL"),
    ),
    TypeInferCase(
        "sub_partial_sum_broadcast_errors",
        _SUB,
        (_PSUM, _BCAST),
        ExpectedError(match="Binary SUB"),
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


# ── lower-rank split right-aligns to the output axis ─────────────────────
# Binary derives the output ShardLayout from the shard-propagation engine
# (mesh-axis bindings), not by carrying a hand-picked layout literal, so these
# check which mesh axis holds Split on the (right-aligned) output axis, not
# the internal layout position count a valid derivation happens to produce.


def test_lower_rank_rhs_split_right_aligns():
    lhs = make_tensor_type((4, 8), _F)
    rhs = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
    out = infer_call(_ADD, lhs, rhs)
    assert out.shape == (4, 8)
    assert out.layout.attrs == (Split(1),)


def test_lower_rank_lhs_split_right_aligns():
    lhs = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
    rhs = make_tensor_type((4, 8), _F)
    out = infer_call(_ADD, lhs, rhs)
    assert out.shape == (4, 8)
    assert out.layout.attrs == (Split(1),)


@pytest.mark.parametrize(
    "kind",
    [BinaryKind.ADD, BinaryKind.SUB, BinaryKind.MUL],
    ids=["add", "sub", "mul"],
)
def test_binary_evaluate(kind):
    torch.manual_seed(0)
    _a, _b = torch.randn(2, 3), torch.randn(2, 3)
    expected = {BinaryKind.ADD: _a + _b, BinaryKind.SUB: _a - _b, BinaryKind.MUL: _a * _b}[kind]
    run_eval_case(EvalCase("", Binary(kind=kind), (_a, _b), expected))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_binary_evaluate_dtypes(dtype):
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, dtype=dtype), torch.randn(2, 3, dtype=dtype)
    run_eval_case(EvalCase("", Binary(kind=BinaryKind.ADD), (a, b), a + b))


# Low-precision dtypes are legal typeinfer operands: inference is purely
# logical, so they pass through like any other element type.
@pytest.mark.parametrize(
    "dt", [DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1], ids=lambda d: d.name
)
def test_binary_low_precision_typeinfer_passthrough(dt):
    run_typeinfer_case(
        TypeInferCase(
            f"low_precision_{dt.name}",
            _ADD,
            (make_tensor_type((4, 8), dt), make_tensor_type((4, 8), dt)),
            make_tensor_type((4, 8), dt),
        )
    )


# ── minimum / maximum surface aliases (asymmetric clamp oracles) ─────────────


@func
def _min_clamp(g: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    return tf.minimum(g, 10.0)


@func
def _asym_clamp(u: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    return tf.maximum(tf.minimum(u, 10.0), -10.0)


def test_min_clamp_matches_torch():
    """``minimum(g, 10)`` == ``torch.clamp(g, max=10)``."""
    torch.manual_seed(0)
    g = torch.randn(4, 256) * 20.0
    out = evaluate(_min_clamp, g, device="cpu")
    torch.testing.assert_close(out.float(), torch.clamp(g, max=10.0), atol=1e-6, rtol=1e-6)


def test_asym_clamp_matches_torch():
    """``maximum(minimum(u, 10), -10)`` == ``torch.clamp(u, -10, 10)``."""
    torch.manual_seed(1)
    u = torch.randn(4, 256) * 20.0
    out = evaluate(_asym_clamp, u, device="cpu")
    torch.testing.assert_close(
        out.float(), torch.clamp(u, min=-10.0, max=10.0), atol=1e-6, rtol=1e-6
    )


def test_minimum_maximum_resolve_to_binary_min_max():
    """``minimum`` / ``maximum`` are surface aliases of the ``Binary`` MIN / MAX
    kinds."""
    lo = _min_clamp.body
    assert isinstance(lo, Call) and isinstance(lo.target, Binary)
    assert lo.target.kind is BinaryKind.MIN

    hi = _asym_clamp.body
    assert isinstance(hi, Call) and isinstance(hi.target, Binary)
    assert hi.target.kind is BinaryKind.MAX
    inner = hi.args[0]
    assert isinstance(inner, Call) and isinstance(inner.target, Binary)
    assert inner.target.kind is BinaryKind.MIN
