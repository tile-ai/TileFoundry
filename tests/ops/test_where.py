"""Where's broadcast, placement-anchor, and causal-selection semantics."""

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
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, run_typeinfer_case
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TrafficBytes

_MESH = make_mesh((4,))


def test_where_evaluates_right_aligned_broadcast():
    condition = torch.tensor([True, False]).reshape(2, 1, 1)
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    other = torch.full((1, 3, 1), -1.0)

    run_eval_case(
        EvalCase(
            "broadcast",
            Where(),
            (condition, input_, other),
            torch.where(condition, input_, other),
        )
    )


TYPE_CASES = [
    TypeInferCase(
        "condition_does_not_anchor_storage",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage="gmem"),
            make_tensor_type((2, 8, 8), DType.f32, storage="rmem"),
            make_tensor_type((1, 8, 1), DType.f32, storage="rmem"),
        ),
        make_tensor_type(
            (2, 8, 8),
            DType.f32,
            layout=Layout((2, 8, 8), (64, 8, 1)),
            storage="rmem",
        ),
    ),
    TypeInferCase(
        "sharded_scores_anchor_distribution",
        Where(),
        (
            make_tensor_type((8, 8), DType.bool, storage=StorageKind.UMAT),
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"
            ),
            make_shard_tensor_type(
                (2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"
            ),
        ),
        make_shard_tensor_type(
            (2, 16, 8, 8), mesh=_MESH, attrs=(Split(1),), storage="smem"
        ),
    ),
    TypeInferCase(
        "condition_must_be_bool",
        Where(),
        (
            make_tensor_type((2, 3), DType.i64),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        ExpectedError(match="condition must have bool dtype"),
    ),
    TypeInferCase(
        "branches_must_share_dtype",
        Where(),
        (
            make_tensor_type((2, 3), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f16),
        ),
        ExpectedError(match="data branch dtype mismatch"),
    ),
    TypeInferCase(
        "all_inputs_must_broadcast",
        Where(),
        (
            make_tensor_type((2, 4), DType.bool),
            make_tensor_type((2, 3), DType.f32),
            make_tensor_type((2, 3), DType.f32),
        ),
        ExpectedError(match="cannot broadcast"),
    ),
]


@pytest.mark.parametrize("case", TYPE_CASES, ids=lambda case: case.name)
def test_where_typeinfer(case):
    run_typeinfer_case(case)


def test_where_cost_counts_selection_and_materialization():
    run_cost_case(
        CostCase(
            "where",
            Where(),
            (
                make_tensor_type((2, 3, 4), DType.bool),
                make_tensor_type((2, 3, 4), DType.f32),
                make_tensor_type((2, 3, 4), DType.f32),
            ),
            flops={DType.bool: 24},
            traffic=(
                TrafficBytes(read=3),
                TrafficBytes(read=96),
                TrafficBytes(read=96),
                TrafficBytes(write=96),
            ),
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
