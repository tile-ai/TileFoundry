"""Cover online GQA specialization and CTA distribution strategies.

Each variant and its dispatch prototype must match the torch decode reference.
A nonaligned context length fails closed instead of dropping a split tail.

See [hir §2](docs/spec/hir.md#2-function-specialization-api).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError

GQA_DECODE = Path(__file__).parents[2] / "fixtures" / "placed" / "gqa_decode.py"


def _constants(*names):
    namespace = load_namespace(str(GQA_DECODE))[0]
    return tuple(namespace[name] for name in names)


def _heads():
    """Query heads, KV heads, head dim, and group size, as the fixture states them."""
    return _constants("NUM_Q_HEADS", "NUM_KV_HEADS", "HEAD_DIM", "GQA_GROUP")


def _selected(name):
    gqa_online_attend = load_authored_ir(f"{GQA_DECODE}:GqaOnline").entry_function()
    head_variant, ctx_variant = gqa_online_attend.variants
    return {"head": head_variant, "splitkv": ctx_variant, "prototype": gqa_online_attend}[name]


def _ref(heads, q, k, v, k_new, v_new):
    """Standard GQA softmax attention over the prior cache plus this step's own position, f32.

    Standard (materialized, non-causal) GQA softmax attention over the prior
    cache plus this step's own position, f32.
    """
    _, _, D, G = heads
    kb = torch.cat([k, k_new], dim=1).repeat_interleave(G, dim=2).float()
    vb = torch.cat([v, v_new], dim=1).repeat_interleave(G, dim=2).float()
    scores = torch.einsum("bshd,bchd->bshc", q.float(), kb) * (1.0 / math.sqrt(D))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bshc,bchd->bshd", probs, vb)


def _inputs(heads, ctx):
    """One decode step: the prior cache of length *ctx*, plus its own K/V."""
    Hq, Hkv, D, _ = heads
    torch.manual_seed(100003 + ctx)
    q = (torch.randn(1, 1, Hq, D) * 0.1).bfloat16()
    k = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    v = (torch.randn(1, ctx, Hkv, D) * 0.1).bfloat16()
    k_new = (torch.randn(1, 1, Hkv, D) * 0.1).bfloat16()
    v_new = (torch.randn(1, 1, Hkv, D) * 0.1).bfloat16()
    return q, k, v, k_new, v_new


EVALUATED = [
    *[pytest.param("head", ctx, id=f"head/{ctx}") for ctx in (0, 1, 37, 256)],
    *[pytest.param("splitkv", ctx, id=f"splitkv/{ctx}") for ctx in (8, 64)],
    *[pytest.param("prototype", ctx, id=f"prototype/{ctx}") for ctx in (0, 64)],
]


@pytest.mark.parametrize(("selected", "ctx"), EVALUATED)
def test_what_is_selected_matches_the_reference(selected, ctx):
    heads = _heads()
    Hq, _, D, _ = heads
    step = _inputs(heads, ctx)
    out = evaluate(_selected(selected), *step)

    assert out.shape == (1, 1, Hq, D)
    assert torch.allclose(out.float(), _ref(heads, *step), atol=2e-2, rtol=2e-2)


def test_context_variant_fails_closed_on_unaligned_ctx():
    (NUM_SPLITS,) = _constants("NUM_SPLITS")
    ctx = NUM_SPLITS + 1
    assert ctx % NUM_SPLITS != 0
    step = _inputs(_heads(), ctx)
    with pytest.raises(EvalError, match=r"op=Reshape: shape .*invalid for input of size"):
        evaluate(_selected("splitkv"), *step)
