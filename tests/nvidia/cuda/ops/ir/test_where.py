"""Where's broadcast, placement-anchor, and causal-selection semantics."""

from __future__ import annotations

import pytest
import torch

from tests.nvidia.cuda.fixtures.placed.prefill_decode_attention import (
    HEAD_DIM,
    HEADS,
    PrefillDecodeAttention,
)
from tilefoundry.evaluator import evaluate


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
    )
    transposed = query.float().transpose(1, 2)
    scores = transposed @ transposed.transpose(-1, -2) / HEAD_DIM**0.5
    future = torch.triu(torch.ones((sequence, sequence), dtype=torch.bool), diagonal=1)
    probabilities = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
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
    )
    queries = query.float().transpose(1, 2)
    keys = key.float().transpose(1, 2).transpose(-1, -2)
    values = value.float().transpose(1, 2)
    expected = torch.softmax(queries @ keys / HEAD_DIM**0.5, dim=-1) @ values
    expected = expected.transpose(1, 2).to(torch.bfloat16)

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
