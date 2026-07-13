"""Low-precision dtype boundary.

fp8e4m3 (canonical), f8e8m0 and f4e2m1 are logical dtype declarations. The
evaluator supports Cast to fp8e4m3 and f8e8m0; f4e2m1 has no evaluator Cast.
Generic arithmetic (Binary / Unary / MatMul / Reduce) rejects all three at
type inference.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.typeinfer_utils import infer_call, ten
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.types import LOW_PRECISION_DTYPES, DType
from tilefoundry.parser.hir_parser import parse_script

_DEV = "cpu"
_LOW_PRECISION_NAMES = ["fp8e4m3", "f8e8m0", "f4e2m1"]

_PRELUDE = (
    "from __future__ import annotations\n"
    "from tilefoundry import func\n"
    "from tilefoundry.dsl.tf import *\n"
    "from tilefoundry.dsl import Tensor\n"
    "\n"
)


def _double_cast_fn(n: int, io_dtype: str, mid_dtype: str):
    """A parsed ``@func`` computing ``cast(cast(x, mid), io)`` over shape ``(n,)``."""
    src = (
        _PRELUDE + "@func\n"
        f'def rt(x: Tensor[({n},), "{io_dtype}"]) -> Tensor[({n},), "{io_dtype}"]:\n'
        f'    return cast(cast(x, "{mid_dtype}"), "{io_dtype}")\n'
    )
    return parse_script(src)


# T1: enum declarations + canonical spelling + grouping set
def test_low_precision_dtypes_declared_and_grouped() -> None:
    for name in _LOW_PRECISION_NAMES:
        assert name in DType.__members__, f"missing DType {name}"
        assert DType[name].value == name
    # fp8e4m3 is the sole canonical fp8 spelling; no alternate is introduced.
    assert "f8e4m3" not in DType.__members__
    assert LOW_PRECISION_DTYPES == frozenset(
        {DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1}
    )


# T2/T3: evaluator Cast round-trips through the torch low-precision dtype
def test_cast_fp8e4m3_double_roundtrip_matches_torch() -> None:
    # Includes fp8e4m3's finite-range boundary (max normal 448.0).
    x = torch.tensor(
        [1.5, 448.0, -448.0, 0.0, 256.0, -3.0, 100.0, 7.0], dtype=torch.bfloat16
    )
    out = evaluate(_double_cast_fn(8, "bf16", "fp8e4m3"), x, device=_DEV)
    ref = x.to(torch.float8_e4m3fn).to(torch.bfloat16)
    torch.testing.assert_close(out, ref)


def test_cast_f8e8m0_double_roundtrip_matches_torch() -> None:
    x = torch.tensor([1.0, 2.0, 4.0, 0.5, 3.0, 100.0], dtype=torch.float32)
    out = evaluate(_double_cast_fn(6, "f32", "f8e8m0"), x, device=_DEV)
    ref = x.to(torch.float8_e8m0fnu).to(torch.float32)
    torch.testing.assert_close(out, ref)


# T4: f4e2m1 has no evaluator Cast (fail-closed via to_torch_dtype)
def test_cast_f4e2m1_has_no_evaluator_support() -> None:
    src = (
        _PRELUDE + "@func\n"
        'def rt(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f4e2m1"]:\n'
        '    return cast(x, "f4e2m1")\n'
    )
    fn = parse_script(src)
    with pytest.raises(EvalError, match=r"unsupported dtype.*f4e2m1"):
        evaluate(fn, torch.randn(4), device=_DEV)


# T5-T8: generic arithmetic rejects every low-precision dtype at typeinfer
def _expect_arith_reject(op, inputs, op_name: str, name: str) -> None:
    with pytest.raises(
        VerifyError,
        match=rf"{op_name}: low-precision dtype {name} is not supported for arithmetic",
    ):
        infer_call(op, *inputs)


@pytest.mark.parametrize("name", _LOW_PRECISION_NAMES)
def test_binary_rejects_low_precision(name) -> None:
    dt = getattr(DType, name)
    _expect_arith_reject(
        Binary(kind=BinaryKind.ADD), (ten((4,), dt), ten((4,), dt)), "Binary", name
    )


@pytest.mark.parametrize("name", _LOW_PRECISION_NAMES)
def test_unary_rejects_low_precision(name) -> None:
    dt = getattr(DType, name)
    _expect_arith_reject(Unary(kind=UnaryKind.NEG), (ten((4,), dt),), "Unary", name)


@pytest.mark.parametrize("name", _LOW_PRECISION_NAMES)
def test_matmul_rejects_low_precision(name) -> None:
    dt = getattr(DType, name)
    _expect_arith_reject(MatMul(), (ten((4, 4), dt), ten((4, 4), dt)), "MatMul", name)


@pytest.mark.parametrize("name", _LOW_PRECISION_NAMES)
def test_reduce_rejects_low_precision(name) -> None:
    dt = getattr(DType, name)
    _expect_arith_reject(
        Reduce(axes=(0,), kind=ReduceKind.SUM), (ten((4,), dt),), "Reduce", name
    )


# Precedence: the low-precision guard runs before ordinary dtype-mismatch / rank
# validation, so a low-precision operand paired with an otherwise-conflicting
# operand still yields the low-precision error (not "dtype mismatch" / rank).
def test_binary_low_precision_takes_precedence_over_dtype_mismatch() -> None:
    _expect_arith_reject(
        Binary(kind=BinaryKind.ADD),
        (ten((4,), DType.fp8e4m3), ten((4,), DType.f32)),
        "Binary",
        "fp8e4m3",
    )


def test_matmul_low_precision_takes_precedence_over_dtype_and_rank() -> None:
    # lhs is low-precision AND rank-1 (would otherwise trip dtype-mismatch then
    # the rank >= 2 check); the low-precision guard must fire first.
    _expect_arith_reject(
        MatMul(),
        (ten((4,), DType.fp8e4m3), ten((4, 4), DType.f32)),
        "MatMul",
        "fp8e4m3",
    )
