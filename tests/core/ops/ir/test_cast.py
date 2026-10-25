"""Cast's sharded layout and its low-precision boundary.

Cast's sharded layout and its low-precision boundary: a sharded input keeps
its ShardLayout (Cast's relation is the identity), fp8 round-trips through the
evaluator, and a sub-byte destination dtype is refused there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core._corpus import load_program
from tests.core.ops.ir.op_case import (
    infer_call,
)
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.types.shard_layout import ShardLayout, Split

FIXTURES = Path(__file__).parents[2] / "fixtures"
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))


def test_cast_carries_sharded_layout():
    sl = ShardLayout(
        layout=Layout(shape=(16, 8), strides=(8, 1)),
        attrs=(Split(axis=0),),
        mesh=_M,
    )
    x = make_tensor_type((16, 8), DType.f32, layout=sl)
    out = infer_call(Cast(dtype=DType.bf16), x)
    assert out.dtype == DType.bf16
    assert out.shape == (16, 8)
    assert out.layout == sl


ROUNDTRIPS = [
    pytest.param(
        "bf16",
        "fp8e4m3",
        torch.bfloat16,
        torch.float8_e4m3fn,
        [1.5, 448.0, -448.0, 0.0, 256.0, -3.0, 100.0, 7.0],
        id="fp8e4m3",
    ),
    pytest.param(
        "f32",
        "f8e8m0",
        torch.float32,
        torch.float8_e8m0fnu,
        [1.0, 2.0, 4.0, 0.5, 3.0, 100.0],
        id="f8e8m0",
    ),
]


@pytest.mark.parametrize(("io_dtype", "mid_dtype", "io_torch", "mid_torch", "values"), ROUNDTRIPS)
def test_a_double_roundtrip_matches_torch(io_dtype, mid_dtype, io_torch, mid_torch, values):
    x = torch.tensor(values, dtype=io_torch)

    function = load_program(FIXTURES / f"logical/cast_{mid_dtype}.py").entry_function()
    out = evaluate(function, x)

    torch.testing.assert_close(out, x.to(mid_torch).to(io_torch))


def test_cast_f4e2m1_has_no_evaluator_support():
    fn = load_program(FIXTURES / "diagnostics/evaluator_cast_f4e2m1.py").entry_function()
    with pytest.raises(EvalError, match=r"unsupported dtype.*f4e2m1"):
        evaluate(fn, torch.randn(4))
