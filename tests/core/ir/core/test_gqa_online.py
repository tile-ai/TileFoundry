"""Target-neutral tests and helpers."""

from __future__ import annotations

import math

import pytest
import torch

from tests.core.fixtures.placed.gqa_decode import (
    GQA_GROUP,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_SPLITS,
    gqa_online_attend,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError

Hq, Hkv, D, G = NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, GQA_GROUP


_SCALE = 1.0 / math.sqrt(D)


_HEAD_VARIANT, _CTX_VARIANT = gqa_online_attend.variants


def _ref(q, k, v, k_new, v_new):
    """Standard GQA softmax attention over the prior cache plus this step's own position, f32.

    Standard (materialized, non-causal) GQA softmax attention over the prior
    cache plus this step's own position, f32.
    """
    kb = torch.cat([k, k_new], dim=1).repeat_interleave(G, dim=2).float()
    vb = torch.cat([v, v_new], dim=1).repeat_interleave(G, dim=2).float()
    scores = torch.einsum("bshd,bchd->bshc", q.float(), kb) * _SCALE
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bshc,bchd->bshd", probs, vb)


def _inputs(ctx):
    """One decode step: the prior cache of length *ctx*, plus its own K/V."""
    torch.manual_seed(100003 + ctx)
    q = (torch.randn(1, 1, Hq, D) * 0.1).bfloat16()
    k = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    v = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    k_new = (torch.randn(1, 1, Hkv, D) * 0.1).bfloat16()
    v_new = (torch.randn(1, 1, Hkv, D) * 0.1).bfloat16()
    return q, k, v, k_new, v_new


EVALUATED = [
    *[pytest.param(_HEAD_VARIANT, ctx, id=f"head/{ctx}") for ctx in (0, 1, 37, 256)],
    *[pytest.param(_CTX_VARIANT, ctx, id=f"splitkv/{ctx}") for ctx in (NUM_SPLITS, NUM_SPLITS * 8)],
    *[pytest.param(gqa_online_attend, ctx, id=f"prototype/{ctx}") for ctx in (0, 64)],
]


@pytest.mark.parametrize(("selected", "ctx"), EVALUATED)
def test_what_is_selected_matches_the_reference(selected, ctx):
    step = _inputs(ctx)
    out = evaluate(selected, *step)

    assert out.shape == (1, 1, Hq, D)
    assert torch.allclose(out.float(), _ref(*step), atol=2e-2, rtol=2e-2)


def test_context_variant_fails_closed_on_unaligned_ctx():
    ctx = NUM_SPLITS + 1
    assert ctx % NUM_SPLITS != 0
    step = _inputs(ctx)
    with pytest.raises(EvalError, match=r"op=Reshape: shape .*invalid for input of size"):
        evaluate(_CTX_VARIANT, *step)
