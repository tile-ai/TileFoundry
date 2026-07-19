"""Flash / online-softmax GQA decode core (`@func` DSL) with context-length
`specialize` and the two CTA-distribution strategies.

Decode regime: query length `seq_len` is a small dynamic dim (1..4); the KV
context length `ctx_len` is the large dynamic dim (designed to 256K) and the
dimension the prototype specializes on. The tests are evaluator-vs-reference
parity (this folder's convention): each variant — and the dispatch prototype —
must compute the same attention as a torch reference. The one non-parity test
is the fail-closed regression for non-split-aligned `ctx_len` (the split-KV
variant must raise, not silently drop the tail).
"""
from __future__ import annotations

import math

import pytest
import torch

from tests.models.qwen3_5_30b_a3b.gqa_online import (
    GQA_GROUP,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_SPLITS,
    gqa_online_attend,
)
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target import CudaTarget

Hq, Hkv, D, G = NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, GQA_GROUP
_SCALE = 1.0 / math.sqrt(D)

# variants[0] = head-on-CTA (small ctx), variants[1] = context-on-CTA (split-KV)
_HEAD_VARIANT, _CTX_VARIANT = gqa_online_attend.variants


def _ref(q, k, v):
    """Standard (materialized, full / non-causal) GQA softmax attention, f32."""
    kb = k.repeat_interleave(G, dim=2).float()  # [1, C, Hq, D]
    vb = v.repeat_interleave(G, dim=2).float()
    scores = torch.einsum("bshd,bchd->bshc", q.float(), kb) * _SCALE  # [1, S, Hq, C]
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bshc,bchd->bshd", probs, vb)  # [1, S, Hq, D]


def _inputs(seq, ctx):
    torch.manual_seed(seq * 100003 + ctx)
    q = (torch.randn(1, seq, Hq, D) * 0.1).bfloat16()
    k = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    v = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    return q, k, v


# ── head-on-CTA: evaluator == reference, any seq × small/mid context ───────


@pytest.mark.parametrize("seq", [1, 2, 4])
@pytest.mark.parametrize("ctx", [1, 8, 37, 256])
def test_head_variant_matches_reference(seq, ctx):
    q, k, v = _inputs(seq, ctx)
    out = evaluate(_HEAD_VARIANT, q, k, v, device="cpu")
    assert out.shape == (1, seq, Hq, D)
    assert torch.allclose(out.float(), _ref(q, k, v), atol=2e-2, rtol=2e-2)


# ── context-on-CTA split-KV: evaluator == reference, any seq × aligned ctx ─
# The context is cut into NUM_SPLITS contiguous blocks by reshape; a
# split-aligned ctx (ctx_len % NUM_SPLITS == 0) reshapes exactly, so the
# two-pass math matches the reference.


@pytest.mark.parametrize("seq", [1, 2, 4])
@pytest.mark.parametrize("ctx", [NUM_SPLITS, NUM_SPLITS * 2, NUM_SPLITS * 8])
def test_context_variant_splitkv_matches_reference(seq, ctx):
    q, k, v = _inputs(seq, ctx)
    out = evaluate(_CTX_VARIANT, q, k, v, device="cpu")
    assert out.shape == (1, seq, Hq, D)
    assert torch.allclose(out.float(), _ref(q, k, v), atol=2e-2, rtol=2e-2)


# ── dispatch prototype: evaluator == reference (small ctx → head-on-CTA) ───


def test_prototype_dispatches_and_matches_reference():
    q, k, v = _inputs(2, 64)
    out = evaluate(gqa_online_attend, q, k, v, device="cpu")
    assert torch.allclose(out.float(), _ref(q, k, v), atol=2e-2, rtol=2e-2)


# ── regression: split-KV fails closed on non-aligned ctx_len ───────────────
# Not an eval==ref test, but the correctness guard for the silent-tail-drop
# bug: the context is split into NUM_SPLITS blocks via a reshape whose block
# length is `ctx_len // NUM_SPLITS`, so a non-aligned ctx_len makes the reshape
# size-mismatch and raise, rather than returning a wrong-but-plausible answer.


@pytest.mark.parametrize("ctx", [NUM_SPLITS + 1, 2 * NUM_SPLITS - 1, 9, 15])
def test_context_variant_fails_closed_on_unaligned_ctx(ctx):
    assert ctx % NUM_SPLITS != 0
    q, k, v = _inputs(2, ctx)
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        evaluate(_CTX_VARIANT, q, k, v, device="cpu")


def _walk_ir(expr, seen=None):
    if seen is None:
        seen = set()
    if expr is None or id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    if isinstance(expr, Call):
        for arg in expr.args:
            yield from _walk_ir(arg, seen)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk_ir(element, seen)
    elif isinstance(expr, GridRegionExpr):
        for arg in expr.init_args:
            yield from _walk_ir(arg, seen)
        yield from _walk_ir(expr.body, seen)
        for value in expr.yield_values:
            yield from _walk_ir(value, seen)


def test_static_fixture_has_one_fixed_online_softmax_region() -> None:
    regions = tuple(
        expr for expr in _walk_ir(qwen_static_online.body) if isinstance(expr, GridRegionExpr)
    )
    assert len(regions) == 1
    region = regions[0]
    assert (region.start, region.extent, region.step) == (0, 4096, 1)
    assert {value.name for value in region.carried_args} == {"m", "l", "o"}
    assert qwen_static_online.target == CudaTarget()
    assert tuple(
        (topology.name, topology.size) for topology in qwen_static_online.topologies
    ) == (("cta", 132),)

    reparsed = parse_script(as_script(qwen_static_online))
    reparsed_regions = tuple(
        expr for expr in _walk_ir(reparsed.body) if isinstance(expr, GridRegionExpr)
    )
    assert len(reparsed_regions) == 1
    assert reparsed_regions[0].extent == 4096
