"""RoPE typeinfer: returns the rotated (q, k); head_dim must be even and match."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_BF = DType.bf16
_M = mesh((4,))


def _rope_inputs(q_shape, k_shape, *, q=None, k=None):
    """The (q, k, cos, sin, pos) input types for a RoPE call. ``q``/``k``
    override the q/k TensorType (e.g. to carry a ShardLayout)."""
    return (
        q if q is not None else ten(q_shape, _BF),
        k if k is not None else ten(k_shape, _BF),
        ten((4096, q_shape[-1]), _BF),
        ten((4096, q_shape[-1]), _BF),
        ten((1,), DType.i32),
    )


CASES = [
    TypeInferCase(
        "returns_rotated_q_k",
        RoPE(),
        _rope_inputs((1, 32, 128), (1, 4, 128)),
        TupleType(fields=(ten((1, 32, 128), _BF), ten((1, 4, 128), _BF))),
    ),
    TypeInferCase(
        "odd_head_dim",
        RoPE(),
        _rope_inputs((1, 32, 127), (1, 4, 127)),
        ExpectedError(match="head_dim 127 must be even", exc=TypeError),
    ),
    TypeInferCase(
        "mismatched_head_dims",
        RoPE(),
        _rope_inputs((1, 32, 128), (1, 4, 64)),
        ExpectedError(match="!= k head_dim", exc=TypeError),
    ),
    # q_out = q*cos + rotate_half(q)*sin is linear in q: commutes with
    # Partial(sum), not Partial(max)/Partial(min) (rotate_half's sign flip
    # breaks monotonicity).
    TypeInferCase(
        "partial_sum_q_passes",
        RoPE(),
        _rope_inputs(
            (1, 32, 128), (1, 4, 128),
            q=sharded((1, 32, 128), (Partial("sum"),), _M, dtype=_BF),
        ),
        TupleType(
            fields=(
                sharded((1, 32, 128), (Partial("sum"),), _M, dtype=_BF),
                ten((1, 4, 128), _BF),
            )
        ),
    ),
    TypeInferCase(
        "partial_max_q_errors",
        RoPE(),
        _rope_inputs(
            (1, 32, 128), (1, 4, 128),
            q=sharded((1, 32, 128), (Partial("max"),), _M, dtype=_BF),
        ),
        ExpectedError(match="RoPE", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rope_typeinfer(case):
    run_typeinfer_case(case)


_TORCH_DTYPE = {torch.float32: DType.f32, torch.int64: DType.i64, torch.int32: DType.i32}


def _ttype(t):
    return TensorType(
        shape=tuple(t.shape), dtype=_TORCH_DTYPE[t.dtype], layout=None, storage="gmem"
    )


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_rope_evaluate():
    """RoPE over a [batch, seq, head, head_dim] layout: cos/sin gathered per
    token by ``pos_ids`` and applied as q*cos + rotate_half(q)*sin."""
    torch.manual_seed(0)
    seq, head_dim, max_pos = 3, 8, 16
    q = torch.randn(1, seq, 4, head_dim)
    k = torch.randn(1, seq, 2, head_dim)
    cos_cache = torch.randn(max_pos, head_dim)
    sin_cache = torch.randn(max_pos, head_dim)
    pos = torch.tensor([5, 6, 7], dtype=torch.int64)

    cos = cos_cache[pos][None, :, None, :]
    sin = sin_cache[pos][None, :, None, :]
    q_ref = q * cos + _rotate_half(q) * sin
    k_ref = k * cos + _rotate_half(k) * sin

    inputs = (q, k, cos_cache, sin_cache, pos)
    params = tuple(Var(type=_ttype(t), name=f"x{i}") for i, t in enumerate(inputs))
    call = Call(type=params[0].type, target=RoPE(), args=params)
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(
        name="rope_case", params=params, body=call, return_type=result_type
    )
    q_out, k_out = evaluate(fn, *inputs, device="cpu")
    torch.testing.assert_close(q_out.float(), q_ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=1e-5, rtol=1e-5)
