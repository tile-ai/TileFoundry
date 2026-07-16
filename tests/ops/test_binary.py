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
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split

_ADD = Binary(kind=BinaryKind.ADD)
_MUL = Binary(kind=BinaryKind.MUL)
_SUB = Binary(kind=BinaryKind.SUB)
_F = DType.f32

# A single-axis mesh (g=4) for flat shards and a two-axis mesh (a=2, b=4) for
# factorized shards; cases reuse these so no test hand-builds a Mesh.
_M = mesh((4,))
_MAB = mesh((2, 4), ("a", "b"))

_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)
_BCAST = ten((16, 8), _F)
_PSUM_AXIS0 = sharded((16, 8), (Partial("sum"), Broadcast()), _MAB)
_PSUM_AXIS1 = sharded((16, 8), (Broadcast(), Partial("sum")), _MAB)

CASES = [
    # ── shape inference (unsharded) ──────────────────────────────────────────
    TypeInferCase("same_shape", _ADD, (ten((4, 8), _F), ten((4, 8), _F)), ten((4, 8), _F)),
    TypeInferCase("size1_broadcast", _ADD, (ten((4, 8), _F), ten((1, 8), _F)), ten((4, 8), _F)),
    TypeInferCase("different_rank_broadcast", _ADD, (ten((4, 8), _F), ten((8,), _F)), ten((4, 8), _F)),
    TypeInferCase("scalar_broadcast", _ADD, (ten((), _F), ten((4, 8), _F)), ten((4, 8), _F)),
    TypeInferCase(
        "dynamic_dim",
        _ADD,
        (ten((DimVar("N", 1, 64), 8), _F), ten((DimVar("N", 1, 64), 8), _F)),
        ten((DimVar("N", 1, 64), 8), _F),
    ),
    TypeInferCase(
        "dtype_mismatch",
        _ADD,
        (ten((4, 8), _F), ten((4, 8), DType.bf16)),
        ExpectedError(match="dtype mismatch"),
    ),
    # ── shard propagation ────────────────────────────────────────────────────
    # lhs split, rhs replicated → output keeps lhs's split.
    TypeInferCase(
        "sharded_lhs_replicated_rhs",
        _ADD,
        (sharded((16, 8), (Split(0),), _M), ten((16, 8), _F)),
        sharded((16, 8), (Split(0),), _M),
    ),
    # both split the same axis identically → that split.
    TypeInferCase(
        "both_split_same_axis",
        _ADD,
        (sharded((16, 8), (Split(0),), _M), sharded((16, 8), (Split(0),), _M)),
        sharded((16, 8), (Split(0),), _M),
    ),
    # split side + broadcast side: lhs (4,8) split axis 0, rhs (8,) broadcasts.
    TypeInferCase(
        "split_side_plus_broadcast_side",
        _ADD,
        (sharded((4, 8), (Split(0),), _M), ten((8,), _F)),
        sharded((4, 8), (Split(0),), _M),
    ),
    # lower-rank split rhs / lhs right-aligns to output axis 1.
    TypeInferCase(
        "lower_rank_rhs_split",
        _ADD,
        (ten((4, 8), _F), sharded((8,), (Split(0),), _M)),
        sharded((4, 8), (Split(1),), _M),
    ),
    TypeInferCase(
        "lower_rank_lhs_split",
        _ADD,
        (sharded((8,), (Split(0),), _M), ten((4, 8), _F)),
        sharded((4, 8), (Split(1),), _M),
    ),
    # lhs splits axis 0, rhs splits axis 1 on the same mesh axis → conflict,
    # not a silent lhs pick.
    TypeInferCase(
        "incompatible_split",
        _ADD,
        (sharded((16, 8), (Split(0),), _M), sharded((16, 8), (Split(1),), _M)),
        ExpectedError(match="incompatible"),
    ),
    TypeInferCase(
        "partial_sum_different_mesh_axes_errors",
        _ADD,
        (_PSUM_AXIS0, _PSUM_AXIS1),
        ExpectedError(match="mesh axis 0"),
    ),
    # two mesh axes split the same tensor axis (neither operand supplies both):
    # the output factorizes axis 0 into one sub-position per mesh extent.
    TypeInferCase(
        "two_mesh_axes_synthesize_factorized",
        _ADD,
        (
            sharded((8,), (Split(0), Broadcast()), _MAB),
            sharded((8,), (Broadcast(), Split(0)), _MAB),
        ),
        sharded((8,), (Split(0), Split(1)), _MAB, cute=(2, 4)),
    ),
    # one operand already carries the full factorized layout → carried through.
    TypeInferCase(
        "factorized_input_passes_through",
        _ADD,
        (sharded((8,), (Split(0), Split(1)), _MAB, cute=(2, 4)), ten((8,), _F)),
        sharded((8,), (Split(0), Split(1)), _MAB, cute=(2, 4)),
    ),
    # ── output storage (anchor on concrete residency) ────────────────────────
    # An unmaterialized literal operand (storage=umat) abstains; the concrete
    # operand anchors the output, independent of operand order.
    TypeInferCase(
        "literal_rhs_anchors_gmem",
        _ADD,
        (ten((4, 8), _F, storage="gmem"), ten((), _F, storage=StorageKind.UMAT)),
        ten((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "literal_lhs_anchors_gmem",
        _ADD,
        (ten((), _F, storage=StorageKind.UMAT), ten((4, 8), _F, storage="gmem")),
        ten((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "both_gmem",
        _ADD,
        (ten((4, 8), _F, storage="gmem"), ten((4, 8), _F, storage="gmem")),
        ten((4, 8), _F, storage="gmem"),
    ),
    TypeInferCase(
        "both_rmem",
        _ADD,
        (ten((4, 8), _F, storage="rmem"), ten((4, 8), _F, storage="rmem")),
        ten((4, 8), _F, storage="rmem"),
    ),
    # All operands unmaterialized (e.g. `1 + 1`) → output stays unmaterialized.
    TypeInferCase(
        "all_unmaterialized",
        _ADD,
        (ten((), _F, storage=StorageKind.UMAT), ten((), _F, storage=StorageKind.UMAT)),
        ten((), _F, storage=StorageKind.UMAT),
    ),
    # Two different concrete residencies have no anchor → error, not a pick.
    TypeInferCase(
        "conflicting_concrete_storage",
        _ADD,
        (ten((4, 8), _F, storage="gmem"), ten((4, 8), _F, storage="rmem")),
        ExpectedError(match="conflicting storage"),
    ),
    # ── Partial(R) commutation ────────────────────────────────────────────────
    # ADD(Partial(sum), Partial(sum)) is sound: sum_x + sum_y == sum(x + y).
    TypeInferCase("add_partial_sum_partial_sum_passes", _ADD, (_PSUM, _PSUM), _PSUM),
    # ADD(Partial(max), Partial(max)) is nonsensical: max(x)+max(y) != max(x+y).
    TypeInferCase(
        "add_partial_max_partial_max_errors", _ADD, (_PMAX, _PMAX),
        ExpectedError(match="Binary ADD"),
    ),
    # ADD(Partial(sum), Broadcast) is the pinned bug: sum(x)+b != sum(x+b).
    TypeInferCase(
        "add_partial_sum_broadcast_errors", _ADD, (_PSUM, _BCAST),
        ExpectedError(match="Binary ADD"),
    ),
    # ADD(Partial(max), Broadcast) is sound: max(x)+b == max(x+b).
    TypeInferCase("add_partial_max_broadcast_passes", _ADD, (_PMAX, _BCAST), _PMAX),
    # MUL(Partial(sum), Broadcast) is sound: b*sum(x) == sum(b*x).
    TypeInferCase("mul_partial_sum_broadcast_passes", _MUL, (_PSUM, _BCAST), _PSUM),
    # MUL(Partial(max), Broadcast) is unsound: b's sign is not provable.
    TypeInferCase(
        "mul_partial_max_broadcast_errors", _MUL, (_PMAX, _BCAST),
        ExpectedError(match="Binary MUL"),
    ),
    # SUB is not in the finalized table: default-reject any Partial operand.
    TypeInferCase(
        "sub_partial_sum_broadcast_errors", _SUB, (_PSUM, _BCAST),
        ExpectedError(match="Binary SUB"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_binary_typeinfer(case):
    run_typeinfer_case(case)


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
    "dt", [DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1], ids=lambda d: d.value
)
def test_binary_low_precision_typeinfer_passthrough(dt):
    run_typeinfer_case(
        TypeInferCase(
            f"low_precision_{dt.value}",
            _ADD,
            (ten((4, 8), dt), ten((4, 8), dt)),
            ten((4, 8), dt),
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
