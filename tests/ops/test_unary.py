"""Unary typeinfer: shape / dtype / layout / storage pass through the input,
including a sharded input's ``ShardLayout``.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
    tensor_grid,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout, Split

_NEG = Unary(kind=UnaryKind.NEG)
_NOT = Unary(kind=UnaryKind.NOT)
_EXP = Unary(kind=UnaryKind.EXP)
_ABS = Unary(kind=UnaryKind.ABS)
_RSQRT = Unary(kind=UnaryKind.RSQRT)

_M = mesh((4,))
_PSUM = sharded((16, 8), (Partial("sum"),), _M)
_PMAX = sharded((16, 8), (Partial("max"),), _M)

CASES_PARTIAL = [
    # NEG is linear: commutes with sum, not max/min.
    TypeInferCase("neg_partial_sum_passes", _NEG, (_PSUM,), _PSUM),
    TypeInferCase(
        "neg_partial_max_errors", _NEG, (_PMAX,),
        ExpectedError(match="Unary NEG"),
    ),
    # EXP/LOG/RELU are monotone-increasing: commute with max/min, not sum.
    TypeInferCase("exp_partial_max_passes", _EXP, (_PMAX,), _PMAX),
    TypeInferCase(
        "exp_partial_sum_errors", _EXP, (_PSUM,),
        ExpectedError(match="Unary EXP"),
    ),
    # ABS/SQUARE/RSQRT/NOT are not proven to commute with any reduction.
    TypeInferCase(
        "abs_partial_sum_errors", _ABS, (_PSUM,),
        ExpectedError(match="Unary ABS"),
    ),
    TypeInferCase(
        "rsqrt_partial_max_errors", _RSQRT, (_PMAX,),
        ExpectedError(match="Unary RSQRT"),
    ),
]


CASES = [
    TypeInferCase(name="passthrough", op=_NEG, inputs=(t,), expected=t)
    for t in tensor_grid((4, 8), DType.f32)
] + [
    TypeInferCase(
        name="not_requires_bool",
        op=_NOT,
        inputs=(ten((4, 8), DType.f32),),
        expected=ExpectedError(match="bool"),
    ),
] + [
    # Low-precision dtypes are legal typeinfer operands: inference is purely
    # logical, so they pass through like any other element type.
    TypeInferCase(
        name=f"low_precision_passthrough_{dt.value}",
        op=_NEG,
        inputs=(ten((4, 8), dt),),
        expected=ten((4, 8), dt),
    )
    for dt in (DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1)
] + CASES_PARTIAL


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_unary_typeinfer(case):
    run_typeinfer_case(case)


def test_unary_passes_sharded_layout_through():
    mesh = Mesh(
        topology="gpu",
        layout=Layout(shape=(4,), strides=(1,)),
        names=("g",),
        topologies=("gpu",),
    )
    sl = ShardLayout(
        layout=Layout(shape=(16, 8), strides=(8, 1)),
        attrs=(Split(0),),
        mesh=mesh,
    )
    x = TensorType(shape=(16, 8), dtype=DType.f32, layout=sl, storage="gmem")
    out = infer_call(_NEG, x)
    assert out.layout is sl
    assert out.shape == (16, 8)


@pytest.mark.parametrize(
    "kind,ref",
    [
        (UnaryKind.NEG, lambda x: -x),
        (UnaryKind.ABS, lambda x: x.abs()),
        (UnaryKind.SQUARE, lambda x: x.square()),
        (UnaryKind.EXP, lambda x: x.exp()),
    ],
    ids=["neg", "abs", "square", "exp"],
)
def test_unary_evaluate(kind, ref):
    torch.manual_seed(0)
    x = torch.randn(4)
    run_eval_case(EvalCase(kind.name.lower(), Unary(kind=kind), (x,), ref(x)))


def test_unary_evaluate_log_positive():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("log", Unary(kind=UnaryKind.LOG), (x,), x.log(), atol=1e-6))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_unary_evaluate_dtypes(dtype):
    torch.manual_seed(0)
    x = torch.randn(4, dtype=dtype)
    run_eval_case(EvalCase("exp", Unary(kind=UnaryKind.EXP), (x,), torch.exp(x)))


# ── exp / log surface resolution and composition oracle ─────────────────────


@func
def _exp_only(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.exp(x)


@func
def _log_only(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.log(x)


@pytest.mark.parametrize(
    "fn,kind", [(_exp_only, UnaryKind.EXP), (_log_only, UnaryKind.LOG)], ids=["exp", "log"]
)
def test_exp_log_resolve_to_unary_kinds(fn, kind):
    """``exp`` / ``log`` are surface aliases of the kinded ``Unary`` op."""
    body = fn.body
    assert isinstance(body, Call) and isinstance(body.target, Unary)
    assert body.target.kind is kind


@func
def _sqrt_softplus(x: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    # softplus(x) = log(1 + exp(x)); sqrt(y) = y * rsqrt(y)
    sp = tf.log(tf.add(tf.exp(x), 1.0))
    return tf.mul(sp, tf.rsqrt(sp))


def test_sqrt_softplus_composition_matches_torch():
    """``sqrt(softplus(x))`` built from ``log`` / ``exp`` / ``rsqrt`` matches
    torch on ``[4, 256] f32``."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    out = evaluate(_sqrt_softplus, x, device="cpu")
    ref = torch.sqrt(torch.nn.functional.softplus(x))
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)
