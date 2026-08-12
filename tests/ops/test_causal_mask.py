"""CausalMask value semantics and shard-sound type contract."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.fixtures.placed.prefill_decode_attention import (
    HEAD_DIM,
    HEADS,
    PrefillDecodeAttention,
)
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.nn.causal_mask import CausalMask
from tilefoundry.ir.types import DType, TensorType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, Split
from tilefoundry.visitor_registry.contexts import TrafficBytes

_START = TensorType.meta_scalar()
_MESH = make_mesh((4,))


def _reference(scores, query_start, key_start, value):
    query = torch.arange(scores.shape[-2]) + int(query_start)
    key = torch.arange(scores.shape[-1]) + int(key_start)
    keep = key.unsqueeze(0) <= query.unsqueeze(1)
    return torch.where(keep, scores, torch.full_like(scores, value))


@pytest.mark.parametrize(
    ("query_start", "key_start"),
    [(0, 0), (4, 0), (0, 4), (2, 1)],
)
def test_causal_mask_evaluate(query_start, key_start):
    scores = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    q_start = torch.tensor(query_start, dtype=torch.int64)
    k_start = torch.tensor(key_start, dtype=torch.int64)
    run_eval_case(
        EvalCase(
            "causal_mask",
            CausalMask(value=-99.0),
            (scores, q_start, k_start),
            _reference(scores, query_start, key_start, -99.0),
        )
    )


def test_causal_mask_default_writes_negative_infinity():
    scores = torch.ones((2, 3), dtype=torch.float32)
    run_eval_case(
        EvalCase(
            "causal_mask_default",
            CausalMask(),
            (
                scores,
                torch.tensor(0, dtype=torch.int64),
                torch.tensor(3, dtype=torch.int64),
            ),
            torch.full_like(scores, float("-inf")),
        )
    )


@pytest.mark.parametrize("sequence", [128, 130])
def test_prefill_matches_full_causal_attention_with_and_without_a_tail(sequence):
    torch.manual_seed(0)
    query = torch.randn((1, sequence, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    empty_cache = torch.zeros((1, 1, HEADS, HEAD_DIM), dtype=torch.bfloat16)

    actual = evaluate(
        PrefillDecodeAttention.entry_function(),
        query,
        empty_cache,
        empty_cache,
        device="cpu",
    )
    transposed = query.float().transpose(1, 2)
    scores = transposed @ transposed.transpose(-1, -2) / HEAD_DIM**0.5
    future = torch.triu(
        torch.ones((sequence, sequence), dtype=torch.bool), diagonal=1
    )
    probabilities = torch.softmax(
        scores.masked_fill(future, float("-inf")), dim=-1
    )
    expected = probabilities @ transposed
    expected = expected.transpose(1, 2).to(torch.bfloat16)

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("context", [128, 130])
def test_decode_matches_full_cache_attention_with_and_without_a_tail(context):
    torch.manual_seed(0)
    query = torch.randn((1, 1, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.randn((1, context, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    value = torch.randn((1, context, HEADS, HEAD_DIM), dtype=torch.bfloat16)

    actual = evaluate(
        PrefillDecodeAttention.entry_function(),
        query,
        key,
        value,
        device="cpu",
    )
    queries = query.float().transpose(1, 2)
    keys = key.float().transpose(1, 2).transpose(-1, -2)
    values = value.float().transpose(1, 2)
    expected = torch.softmax(queries @ keys / HEAD_DIM**0.5, dim=-1) @ values
    expected = expected.transpose(1, 2).to(torch.bfloat16)

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)


TYPEINFER_CASES = [
    TypeInferCase(
        "head_split_preserved",
        CausalMask(),
        (
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),)
            ),
            _START,
            _START,
        ),
        make_shard_tensor_type((2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),)),
    ),
    TypeInferCase(
        "query_split_rejected",
        CausalMask(),
        (
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Split(2),)
            ),
            _START,
            _START,
        ),
        ExpectedError(match="query axis"),
    ),
    TypeInferCase(
        "key_split_rejected",
        CausalMask(),
        (
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Split(3),)
            ),
            _START,
            _START,
        ),
        ExpectedError(match="key axis"),
    ),
    TypeInferCase(
        "partial_rejected",
        CausalMask(),
        (
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Partial("sum"),)
            ),
            _START,
            _START,
        ),
        ExpectedError(match="CausalMask"),
    ),
    TypeInferCase(
        "rank_one_scores_rejected",
        CausalMask(),
        (make_tensor_type((8,)), _START, _START),
        ExpectedError(match="rank >= 2"),
    ),
    TypeInferCase(
        "non_scalar_start_rejected",
        CausalMask(),
        (
            make_tensor_type((2, 3)),
            make_tensor_type((1,), DType.i64),
            _START,
        ),
        ExpectedError(match="query_start must be a rank-0 integer"),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda case: case.name)
def test_causal_mask_typeinfer(case):
    run_typeinfer_case(case)


def test_causal_mask_cost_counts_one_predicate_per_score():
    run_cost_case(
        CostCase(
            "causal_mask",
            CausalMask(),
            (
                make_tensor_type((2, 3, 4), DType.f32),
                _START,
                _START,
            ),
            flops={DType.bool: 24},
            traffic=(
                TrafficBytes(read=96),
                TrafficBytes(read=8),
                TrafficBytes(read=8),
                TrafficBytes(write=96),
            ),
        )
    )
