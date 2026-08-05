"""Cast's sharded layout and its low-precision boundary: a sharded input keeps
its ShardLayout (Cast's relation is the identity), fp8 round-trips through the
evaluator, and a sub-byte destination dtype is refused there."""
from __future__ import annotations

import pytest
import torch

from tests.ops.typeinfer_utils import (
    infer_call,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tilefoundry.parser.hir_parser import parse_script

_M = make_mesh((4,))


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
    assert out.layout == sl  # identity relation -> same ShardLayout


# ── low-precision Cast boundary (fp8e4m3 / f8e8m0 evaluator; f4e2m1 declared) ──

_PRELUDE = (
    "from __future__ import annotations\n"
    "from tilefoundry import func\n"
    "from tilefoundry.dsl import Tensor, tf\n"
    "\n"
)


def _double_cast_fn(n: int, io_dtype: str, mid_dtype: str):
    """A parsed ``@func`` computing ``cast(cast(x, mid), io)`` over shape ``(n,)``."""
    src = (
        _PRELUDE + "@func\n"
        f'def rt(x: Tensor[({n},), "{io_dtype}"]) -> Tensor[({n},), "{io_dtype}"]:\n'
        f'    return tf.cast(tf.cast(x, "{mid_dtype}"), "{io_dtype}")\n'
    )
    return parse_script(src)


#: The low-precision dtypes the evaluator does support, each against torch's own.
#: fp8e4m3's values include its finite-range boundary, max normal 448.0.
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

    out = evaluate(_double_cast_fn(len(values), io_dtype, mid_dtype), x, device="cpu")

    torch.testing.assert_close(out, x.to(mid_torch).to(io_torch))


def test_cast_f4e2m1_has_no_evaluator_support():
    src = (
        _PRELUDE + "@func\n"
        'def rt(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f4e2m1"]:\n'
        '    return tf.cast(x, "f4e2m1")\n'
    )
    fn = parse_script(src)
    with pytest.raises(EvalError, match=r"unsupported dtype.*f4e2m1"):
        evaluate(fn, torch.randn(4), device="cpu")
