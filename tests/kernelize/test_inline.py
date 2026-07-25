"""``inline_calls`` -- flatten nested ``Call(target=Function)`` bodies, the
hard prerequisite for ``kernelize.extract`` to walk a whole composed layer
(e.g. Qwen3-1.7B's ``decoder_layer`` = ``self_attention`` + ``mlp``, each of
which nests further ``@func`` calls like ``input_rms_norm`` -- see
``tests/models/qwen3_1_7b/qwen3_1_7b_module.py``).

Three checks, mirroring ``test_gemm_rmsnorm.py`` / ``test_extract_elementwise.py``'s
shape:

1. Structural: ``inline_calls(Qwen3_1_7B.decoder_layer)``'s body has zero
   remaining ``Call(target=Function)`` -- only primitive-op Calls, in
   exactly the op multiset self_attention (27) + mlp (7) + decoder_layer's
   own 2 residual adds compose to.
2. Numerical: inlining is a pure structural rewrite -- the inlined Function
   must evaluate to the *same* output as the original nested one, given the
   same inputs (built the same way ``test_qwen3_1_7b_module.py`` does).
3. Combination: a one-hop wrapper ``@func`` nesting a call to
   ``Qwen3_1_7B.mlp`` -- once flattened, ``extract`` succeeds exactly like
   ``test_extract_qwen3_mlp_whole``'s direct ``Qwen3_1_7B.mlp`` does.
   ``self_attention``/``decoder_layer`` are not extractable yet (RoPE has no
   V1 extract fallback) -- out of scope here, per the task.
"""
from __future__ import annotations

from collections import Counter

import torch

from tests.models.qwen3_1_7b import common
from tests.models.qwen3_1_7b.qwen3_1_7b_module import Qwen3_1_7B
from tilefoundry import func
from tilefoundry.analysis.analyzer import _postorder
from tilefoundry.dsl import Tensor
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.function import Function
from tilefoundry.kernelize import TileGraph, extract, inline_calls

HIDDEN = common.HIDDEN
S_CAP = common.S_CAP
DT = common.DT
INTERMEDIATE = common.INTERMEDIATE
DEV = "cpu"
ATOL = RTOL = 2e-4


def _nested_function_calls(body) -> list[Call]:
    """Every ``Call(target=Function)`` reachable in ``body`` -- the exact
    construct ``kernelize.extract`` rejects and ``inline_calls`` removes."""
    return [e for e in _postorder(body) if isinstance(e, Call) and isinstance(e.target, Function)]


def test_inline_flattens_decoder_layer_to_primitive_ops():
    """``decoder_layer`` nests ``self_attention``/``mlp`` -- after
    ``inline_calls``, none of that remains: every ``Call`` in the flattened
    body targets a primitive op."""
    original_nested = _nested_function_calls(Qwen3_1_7B.decoder_layer.body)
    assert {c.target.name for c in original_nested} == {"self_attention", "mlp"}

    flat = inline_calls(Qwen3_1_7B.decoder_layer)
    assert isinstance(flat, Function)
    assert flat is not Qwen3_1_7B.decoder_layer
    assert _nested_function_calls(flat.body) == []

    order = _postorder(flat.body)
    op_seq = [type(e.target).__name__ for e in order if isinstance(e, Call)]
    print("\n=== decoder_layer (inlined): op sequence (postorder) ===")
    print(op_seq)

    # self_attention's 27 ops (3 RMSNorm, 6 MatMul, 4 Reshape, 2 RoPE,
    # 2 TupleGetItem, 2 RepeatInterleave, 5 Transpose, 2 Binary, 1 SoftMax)
    # + mlp's 7 (1 RMSNorm, 3 MatMul, 1 Sigmoid, 2 Binary)
    # + decoder_layer's own 2 residual-add Binary.
    assert Counter(op_seq) == Counter(
        {
            "RMSNorm": 4,
            "MatMul": 9,
            "Reshape": 4,
            "RoPE": 2,
            "TupleGetItem": 2,
            "RepeatInterleave": 2,
            "Transpose": 5,
            "Binary": 6,
            "SoftMax": 1,
            "Sigmoid": 1,
        }
    )
    assert len(op_seq) == 36


def _decoder_layer_inputs():
    """Same fixture recipe as ``test_decoder_layer_evaluate`` in
    ``test_qwen3_1_7b_module.py``: a fixed-seed HF layer's weights + RoPE
    caches / causal mask / attention scale, plus a random hidden-state input.
    """
    layer = common.build_hf_layer(seed=0, device=DEV)
    cfg = common.build_hf_config()
    cos_cache, sin_cache = common.rope_caches(cfg, S_CAP, device=DEV)
    pos_ids = torch.arange(S_CAP, device=DEV, dtype=torch.int32)
    mask = common.causal_mask(S_CAP, device=DEV)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=DEV)
    attn, mlp = layer.self_attn, layer.mlp

    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    return (
        x,
        layer.input_layernorm.weight,
        common.linear_weight(attn.q_proj),
        common.linear_weight(attn.k_proj),
        common.linear_weight(attn.v_proj),
        attn.q_norm.weight,
        attn.k_norm.weight,
        cos_cache,
        sin_cache,
        pos_ids,
        mask,
        scale,
        common.linear_weight(attn.o_proj),
        layer.post_attention_layernorm.weight,
        common.linear_weight(mlp.gate_proj),
        common.linear_weight(mlp.up_proj),
        common.linear_weight(mlp.down_proj),
    )


def test_inline_preserves_numerics_vs_original():
    """Pure structural rewrite -- inlining must not change what the function
    computes. The same inputs into the original (nested) and inlined (flat)
    ``decoder_layer`` must produce numerically-close output."""
    inputs = _decoder_layer_inputs()

    original_out = evaluate(Qwen3_1_7B.decoder_layer, *inputs, device=DEV)
    inlined_out = evaluate(inline_calls(Qwen3_1_7B.decoder_layer), *inputs, device=DEV)

    torch.testing.assert_close(inlined_out.float(), original_out.float(), atol=ATOL, rtol=RTOL)


_mlp_fn = Qwen3_1_7B.mlp  # bare-name binding -- nested @func calls resolve by name only (parser/base.py)


@func
def _mlp_via_nested_call(
    hidden: Tensor[(1, S_CAP, HIDDEN), DT],
    gamma_post: Tensor[(HIDDEN,), DT],
    w_gate: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
    w_up: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
    w_down: Tensor[(1, INTERMEDIATE, HIDDEN), DT],
) -> Tensor[(1, S_CAP, HIDDEN), DT]:
    # One-hop wrapper around Qwen3_1_7B.mlp -- exercises inline_calls on a
    # nested call shaped exactly like decoder_layer's own `mlp(...)` site,
    # without dragging in self_attention's (not-yet-extractable) subgraph.
    return _mlp_fn(hidden, gamma_post, w_gate, w_up, w_down)


def test_inline_then_extract_mlp_subgraph():
    """``mlp`` called through one nested ``@func`` hop -- once flattened,
    ``extract`` succeeds exactly like ``test_extract_qwen3_mlp_whole``'s
    direct ``Qwen3_1_7B.mlp`` does (same 7-statement shape)."""
    flat = inline_calls(_mlp_via_nested_call)
    assert _nested_function_calls(flat.body) == []

    tg = extract(flat)
    assert isinstance(tg, TileGraph)
    op_names = [type(u.op.target).__name__ for u in tg.units]
    assert Counter(op_names) == Counter({"RMSNorm": 1, "MatMul": 3, "Sigmoid": 1, "Binary": 2})
    assert len(tg.units) == 7
