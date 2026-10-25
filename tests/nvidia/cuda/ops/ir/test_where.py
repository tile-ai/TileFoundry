"""Where's broadcast, placement-anchor, and causal-selection semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.evaluator import evaluate

PREFILL_DECODE = Path(__file__).parents[2] / "fixtures" / "placed" / "prefill_decode_attention.py"


def _attention():
    """The fixture's head count, head width, and authored entry."""
    constants = load_namespace(str(PREFILL_DECODE))[0]
    entry = load_authored_ir(f"{PREFILL_DECODE}:PrefillDecodeAttention").entry_function()
    return constants["HEADS"], constants["HEAD_DIM"], entry


@pytest.mark.parametrize("sequence", [128, 130])
def test_prefill_matches_full_causal_attention_with_and_without_a_tail(sequence):
    HEADS, HEAD_DIM, attend = _attention()
    torch.manual_seed(0)
    query = torch.randn((1, sequence, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    empty_cache = torch.zeros((1, 1, HEADS, HEAD_DIM), dtype=torch.bfloat16)

    actual = evaluate(
        attend,
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
    HEADS, HEAD_DIM, attend = _attention()
    torch.manual_seed(0)
    query = torch.randn((1, 1, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.randn((1, context, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    value = torch.randn((1, context, HEADS, HEAD_DIM), dtype=torch.bfloat16)

    actual = evaluate(
        attend,
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
