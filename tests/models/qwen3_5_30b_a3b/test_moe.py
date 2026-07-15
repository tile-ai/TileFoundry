"""Qwen3-30B-A3B MoE component: bf16 HIR description + evaluator-vs-HF.

The MoE component (built in ``common``) covers post-attention RMSNorm, the gate
(router) projection, ``softmax`` + top-8 expert selection with normalized
routing weights, the per-expert gate/up projection with SiLU, and the
expert-down projection combined by routing weights. Expert selection is runtime
data: the ``topk`` ``indices`` / ``probs`` drive a ``gather`` of the expert
weights and a batched ``matmul`` over the ``[tokens, top_k]`` grid — no static
128-way expansion and no Python control flow. The residual add belongs to the
decoder layer. It is validated against the ``post_attention_layernorm`` +
``mlp`` submodules of a Qwen3-30B-A3B ``Qwen3MoeDecoderLayer``.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_30b_a3b import common
from tilefoundry.evaluator import evaluate

HIDDEN = common.HIDDEN
MOE_INTERMEDIATE = common.MOE_INTERMEDIATE


_COMBOS = [1, 4]
_DTYPES = [("f32", torch.float32, 2e-4, 2e-4), ("bf16", torch.bfloat16, 3e-2, 3e-2)]


@pytest.mark.parametrize("seq", _COMBOS, ids=lambda v: str(v))
@pytest.mark.parametrize(
    "dt_name,torch_dt,atol,rtol", _DTYPES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_moe_matches_hf(dt_name, torch_dt, atol, rtol, seq):
    common.DT = dt_name
    fn = common.build_moe()

    dev = "cuda"
    torch.manual_seed(0)
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    moe = layer.mlp

    x = (torch.randn(1, seq, HIDDEN, device=dev) * 0.1).to(torch_dt)
    with torch.no_grad():
        ref = moe(layer.post_attention_layernorm(x))

    gup = moe.experts.gate_up_proj
    w_gate = gup[:, :MOE_INTERMEDIATE, :].contiguous()
    w_up = gup[:, MOE_INTERMEDIATE:, :].contiguous()
    w_down = moe.experts.down_proj.contiguous()
    w_router = moe.gate.weight.t().contiguous()

    out = evaluate(
        fn,
        x,
        layer.post_attention_layernorm.weight,
        w_router,
        w_gate,
        w_up,
        w_down,
        device=dev,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
