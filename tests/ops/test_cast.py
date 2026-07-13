"""Cast typeinfer: dtype changes; shape / storage / layout pass through. A
sharded input keeps its ShardLayout (Cast's relation is the identity)."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
    ten,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tilefoundry.parser.hir_parser import parse_script


def _mesh() -> Mesh:
    return Mesh(
        topology="gpu",
        layout=Layout(shape=(4,), strides=(1,)),
        names=("g",),
        topologies=("gpu",),
    )


_M = _mesh()

CASES = [
    TypeInferCase(
        name="unsharded_dtype_change",
        op=Cast(dtype=DType.bf16),
        inputs=(ten((4, 8), DType.f32),),
        expected=ten((4, 8), DType.bf16),
    ),
    TypeInferCase(
        name="rank0",
        op=Cast(dtype=DType.f32),
        inputs=(ten((), DType.i32),),
        expected=ten((), DType.f32),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_cast_typeinfer(case):
    run_typeinfer_case(case)


def test_cast_carries_sharded_layout():
    sl = ShardLayout(
        layout=Layout(shape=(16, 8), strides=(8, 1)),
        attrs=(Split(axis=0),),
        mesh=_M,
    )
    x = TensorType(shape=(16, 8), dtype=DType.f32, layout=sl, storage="gmem")
    out = infer_call(Cast(dtype=DType.bf16), x)
    assert out.dtype == DType.bf16
    assert out.shape == (16, 8)
    assert out.layout == sl  # identity relation -> same ShardLayout


def test_cast_evaluate():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    run_eval_case(
        EvalCase("to_bf16", Cast(dtype=DType.bf16), (x,), x.to(torch.bfloat16), atol=2e-2, rtol=2e-2)
    )


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


def test_cast_fp8e4m3_double_roundtrip_matches_torch():
    # Includes fp8e4m3's finite-range boundary (max normal 448.0).
    x = torch.tensor(
        [1.5, 448.0, -448.0, 0.0, 256.0, -3.0, 100.0, 7.0], dtype=torch.bfloat16
    )
    out = evaluate(_double_cast_fn(8, "bf16", "fp8e4m3"), x, device="cpu")
    ref = x.to(torch.float8_e4m3fn).to(torch.bfloat16)
    torch.testing.assert_close(out, ref)


def test_cast_f8e8m0_double_roundtrip_matches_torch():
    x = torch.tensor([1.0, 2.0, 4.0, 0.5, 3.0, 100.0], dtype=torch.float32)
    out = evaluate(_double_cast_fn(6, "f32", "f8e8m0"), x, device="cpu")
    ref = x.to(torch.float8_e8m0fnu).to(torch.float32)
    torch.testing.assert_close(out, ref)


def test_cast_f4e2m1_has_no_evaluator_support():
    src = (
        _PRELUDE + "@func\n"
        'def rt(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f4e2m1"]:\n'
        '    return tf.cast(x, "f4e2m1")\n'
    )
    fn = parse_script(src)
    with pytest.raises(EvalError, match=r"unsupported dtype.*f4e2m1"):
        evaluate(fn, torch.randn(4), device="cpu")
