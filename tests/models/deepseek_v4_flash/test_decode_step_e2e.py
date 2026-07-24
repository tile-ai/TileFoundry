"""DeepSeek V4 flash decode-step E2E — layer-0 real structure (M4+ of
docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md): embed ->
attention (``attn.mla_kv_update_v2`` / ``attn.mla_attend_v2``, real MLA
sliding-window dims) -> MoE (``moe.py``, hash router by default) -> final
RMSNorm -> lm_head. The GQA placeholder this file used before (dynamic
``DimVar`` context, fake shapes) is retired; see ``decode_step.py``'s module
docstring.

Three variants, each comparing (a) the plain HIR evaluator, no leaves
registered, against (b) every leaf (M1) registered with its torch-cuda
``ImplementationPackage`` (``torch_impl.build_full_leaf_registry``):

  1. ``test_decode_step_e2e`` — real-shape (``REAL_*``) **random** weights,
     no checkpoint needed, hash-routed MoE (the default structure) — a fast
     mechanism test, parametrized over a partial-window and a
     wrapped-window decode position. Random KV-cache pre-state is only used
     here (see ``_prefill_kv_cache``'s docstring for why the two real-weight
     variants below build theirs honestly instead).
  2. ``test_decode_step_e2e_layer0_real_weights`` — **layer 0 fully real**:
     attention + hash-routed MoE + embed/norm/lm_head all from the real
     DeepSeek-V4-Flash-FP8 checkpoint. Gate is rel_l2 / cosine on the
     logits (attention is real now, so the decoded token ids are not
     meaningless — the top-5 is still only a sanity signal, not a true
     next-token prediction, since just one of 43 real transformer layers
     runs here).
  3. ``test_decode_step_e2e_learned_moe_real_weights`` — real layer-3
     learned-router (``moe_topk``) MoE weights, kept to retain
     learned-router coverage now that the default structure is hash-routed;
     attention is real-**shape** random (real layer 3's actual attention has
     ``compress_ratio==4`` Compressor/Indexer KV-compression sparsity, which
     ``attention.py`` does not implement in HIR at all — a pre-existing,
     documented scope gap, not something this revision addresses).

Both real-weight variants build their KV-cache pre-state via
``_prefill_kv_cache`` (option (ii) from the task: decode T prior tokens one
at a time through the real ``mla_kv_update_v2`` HIR update, starting from an
empty cache) rather than fabricating a plausible-looking random cache —
"random cache" is only used in variant 1's mechanism test.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from tests.models.deepseek_v4_flash import attention as attn
from tests.models.deepseek_v4_flash import decode_step, hf_weights, torch_impl
from tests.models.deepseek_v4_flash.moe import (
    DIM,
    MOE_INTER,
    N_ACT,
    N_ROUTED,
    combine_expert_outputs,
    moe_hash_gather,
    pre_moe_rms_norm,
    shared_expert,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.leaf import WeightLoader

DEVICE = "cuda"

# Real FP8 checkpoint (see hf_weights.py). Not present on every machine /
# in CI, so the real-weight tests skip (not fail) when it is missing.
CKPT_DIR = Path("/data2/models/DeepSeek-V4-Flash-FP8")
# First learned-router (noaux_tc, gate.weight + gate.bias) real layer —
# config.json num_hash_layers=3 means layers 0..2 use hash routing.
LEARNED_LAYER = 3

# bf16 end-to-end tolerance for the random-weight mechanism test. Every
# matmul (attention low-rank Q/KV/O projections, MoE gate/up/down for 6
# routed experts + the shared expert, lm_head) carries its own ~bf16-ulp
# rounding (~2^-8 relative per op); qwen's own decode-step fixture
# (tests/models/qwen3_5_30b_a3b/test_decode_step.py) uses atol=rtol=2e-2 for
# a *single* decoder layer at bf16. This graph is deeper and
# evaluator-vs-torch-cuda is a second independent execution path on top of
# that (not just a different dtype), so the tolerance is loosened
# proportionally rather than reused as-is. Also reused for the real-weight
# tests' KV-cache comparison (their logits gate is rel_l2/cosine instead,
# see below).
ATOL = 8e-2
RTOL = 8e-2


def _bf16(gen: torch.Generator, *shape: int, scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen, device=DEVICE) * scale).to(torch.bfloat16)


def _fp8e4m3(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)


def _pow2_scale(gen: torch.Generator, *shape: int) -> torch.Tensor:
    """A 128x128-block scale as an exact power of two (2**-2 .. 2**2), f32 —
    moe.py's block-scale ConstTensors are declared "f32" (the real FP8
    checkpoint stores its ue8m0-semantics block scale pre-expanded to plain
    float32 already, see hf_weights.py), and generating an exact power of
    two here mirrors the real checkpoint's own values (verified: e.g.
    2**-12) instead of an arbitrary float that would just be a stand-in."""
    exps = torch.randint(-2, 3, shape, generator=gen, device=DEVICE).float()
    return torch.exp2(exps)


def _random_attention_weights_v2(gen: torch.Generator) -> dict[str, torch.Tensor]:
    """``layer0.attention.*`` real-**shape** (``REAL_*``) random weights —
    used by the fast random-weight mechanism test and by the layer-3
    learned-MoE coverage test (whose attention cannot be real: real layer 3
    has ``compress_ratio==4`` Compressor/Indexer KV-compression sparsity,
    which ``attention.py`` does not implement in HIR at all). Real layer 0's
    attention (``compress_ratio==0``) IS implementable and is loaded for
    real via ``hf_weights.load_layer0_attention_weights`` instead — see
    ``layer0_real_weights`` below."""
    w: dict[str, torch.Tensor] = {}
    w["layer0.attention.gamma_kv"] = _bf16(gen, attn.REAL_HEAD_DIM)
    w["layer0.attention.w_kv"] = _bf16(gen, attn.REAL_DIM, attn.REAL_HEAD_DIM)
    w["layer0.attention.gamma_q_lora"] = _bf16(gen, attn.REAL_Q_LORA_RANK)
    w["layer0.attention.w_q_a"] = _bf16(gen, attn.REAL_DIM, attn.REAL_Q_LORA_RANK)
    w["layer0.attention.w_q_b"] = _bf16(gen, attn.REAL_Q_LORA_RANK, attn.REAL_Q_PROJ)
    w["layer0.attention.attn_sink"] = _bf16(gen, 1, attn.REAL_N_HEADS, 1, 1)
    w["layer0.attention.w_o_a"] = _bf16(gen, attn.REAL_O_GROUPS, attn.REAL_WO_A_IN, attn.REAL_O_LORA_RANK)
    w["layer0.attention.w_o_b"] = _bf16(gen, attn.REAL_WO_A_OUT, attn.REAL_DIM)
    return w


def _random_weights(seed: int) -> dict[str, torch.Tensor]:
    """Full random weight set, module-path-prefixed (M1) — one decoder
    layer's worth, hash-routed MoE (the default/primary real structure) +
    real-shape attention, matching moe.py's real-size N_ROUTED / MOE_INTER /
    DIM."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    w: dict[str, torch.Tensor] = {}

    w["embed.table"] = _bf16(gen, decode_step.VOCAB, DIM)
    w["lm_head.weight"] = _bf16(gen, DIM, decode_step.VOCAB)
    w["final_rms_norm.weight"] = _bf16(gen, DIM)

    w.update(_random_attention_weights_v2(gen))

    w["layer0.moe.rms_weight"] = (torch.randn(DIM, generator=gen, device=DEVICE) * 0.02)
    w["layer0.moe.gate_weight"] = _bf16(gen, N_ROUTED, DIM)
    w["layer0.moe.tid2eid"] = torch.randint(
        0, N_ROUTED, (decode_step.VOCAB, N_ACT), generator=gen, device=DEVICE, dtype=torch.int64
    )
    # Routed experts: real fp8e4m3 weight + 128x128-block f32 scale (same
    # format as the shared expert below and as the real checkpoint).
    w["layer0.moe.routed_w1_weight"] = _fp8e4m3(gen, N_ROUTED, MOE_INTER, DIM)
    w["layer0.moe.routed_w3_weight"] = _fp8e4m3(gen, N_ROUTED, MOE_INTER, DIM)
    w["layer0.moe.routed_w2_weight"] = _fp8e4m3(gen, N_ROUTED, DIM, MOE_INTER)
    w["layer0.moe.routed_w1_scale"] = _pow2_scale(gen, N_ROUTED, MOE_INTER // 128, DIM // 128)
    w["layer0.moe.routed_w3_scale"] = _pow2_scale(gen, N_ROUTED, MOE_INTER // 128, DIM // 128)
    w["layer0.moe.routed_w2_scale"] = _pow2_scale(gen, N_ROUTED, DIM // 128, MOE_INTER // 128)

    # Shared expert: real fp8e4m3 + f32 block-scale weights —
    # shared_expert_post_init (M1) does a genuine dequant of these, once,
    # cached.
    w["layer0.moe.shared_expert.w1_weight"] = _fp8e4m3(gen, MOE_INTER, DIM)
    w["layer0.moe.shared_expert.w3_weight"] = _fp8e4m3(gen, MOE_INTER, DIM)
    w["layer0.moe.shared_expert.w2_weight"] = _fp8e4m3(gen, DIM, MOE_INTER)
    w["layer0.moe.shared_expert.w1_scale"] = _pow2_scale(gen, MOE_INTER // 128, DIM // 128)
    w["layer0.moe.shared_expert.w3_scale"] = _pow2_scale(gen, MOE_INTER // 128, DIM // 128)
    w["layer0.moe.shared_expert.w2_scale"] = _pow2_scale(gen, DIM // 128, MOE_INTER // 128)
    return w


@pytest.fixture(scope="module")
def loader() -> WeightLoader:
    return WeightLoader(decode_step.decode_step_module)


@pytest.fixture(scope="module")
def weights(loader: WeightLoader) -> dict[str, torch.Tensor]:
    """Loaded (post_init'd) once, reused by every test in this module that
    uses it — exercises the post_init cache across every cur_pos / leaf-
    registration combination below, not just once."""
    raw = _random_weights(seed=0)
    return loader.load(raw)


@pytest.fixture(scope="module")
def full_hash_leaves() -> dict:
    return torch_impl.build_full_leaf_registry(decode_step.decode_step_module).by_function_name()


@pytest.fixture(scope="module")
def full_learned_leaves() -> dict:
    return torch_impl.build_full_leaf_registry(decode_step.decode_step_module_learned).by_function_name()


@pytest.fixture(scope="module", autouse=True)
def _warm_cuda(weights, full_hash_leaves):
    """First-ever CUDA call in a process pays context init / kernel-cache /
    allocator warm-up cost — one throwaway decode step (discarded, both
    execution modes) before any *timed* run below keeps the wall times
    informative."""
    kv_cache_prev = torch.zeros(1, attn.REAL_WINDOW, 1, attn.REAL_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    for leaves in (None, full_hash_leaves):
        decode_step.run_decode_step(
            weights,
            token_ids=torch.zeros(1, device=DEVICE, dtype=torch.int64),
            kv_cache_prev=kv_cache_prev,
            cur_pos=3, device=DEVICE, moe_kind="hash", leaves=leaves,
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("cur_pos", [63, 200], ids=["partial_window", "wrapped_window"])
def test_decode_step_e2e(cur_pos, weights, full_hash_leaves, loader):
    """(a) pure evaluator vs (b) every leaf torch-cuda-registered, same
    (random) weights, random KV cache / token — logits and updated KV cache
    agree within tolerance. Fast mechanism test (no real checkpoint needed).
    Parametrized over a partial-window (cur_pos < REAL_WINDOW - 1, the
    attn_mask branch) and a wrapped-window (cur_pos >= REAL_WINDOW, ring-
    buffer overwrite) decode position against the *same* weights."""
    torch.manual_seed(cur_pos)
    token_ids = torch.randint(0, decode_step.VOCAB, (1,), device=DEVICE, dtype=torch.int64)
    kv_cache_prev = (
        torch.randn(1, attn.REAL_WINDOW, 1, attn.REAL_HEAD_DIM, device=DEVICE) * 0.02
    ).to(torch.bfloat16)

    def run(leaves):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = decode_step.run_decode_step(
            weights, token_ids=token_ids, kv_cache_prev=kv_cache_prev, cur_pos=cur_pos,
            device=DEVICE, moe_kind="hash", leaves=leaves,
        )
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    (logits_ref, kv_ref), t_pure = run(None)
    (logits_leaf, kv_leaf), t_leaf = run(full_hash_leaves)

    assert logits_ref.shape == (1, 1, decode_step.VOCAB)
    assert torch.isfinite(logits_ref).all()
    assert torch.isfinite(logits_leaf).all()

    max_err = (logits_ref.float() - logits_leaf.float()).abs().max().item()
    print(
        f"\n[decode_step_e2e cur_pos={cur_pos}] wall time: pure_evaluator={t_pure * 1e3:.2f}ms "
        f"leaf_registered={t_leaf * 1e3:.2f}ms; logits max abs err={max_err:.4g} "
        f"(atol={ATOL}, rtol={RTOL}); post_init_runs so far={loader.post_init_runs}"
    )

    torch.testing.assert_close(logits_leaf.float(), logits_ref.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(kv_leaf.float(), kv_ref.float(), atol=ATOL, rtol=RTOL)


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float().reshape(-1), ref.float().reshape(-1)
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _cosine(got: torch.Tensor, ref: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        got.float().reshape(1, -1), ref.float().reshape(1, -1)
    ).item()


def _prefill_kv_cache(weights: dict[str, torch.Tensor], token_ids_seq: torch.Tensor, device: str) -> torch.Tensor:
    """Sliding-window KV cache after decoding ``token_ids_seq`` (shape (T,))
    one token at a time via ``decode_step.kv_update_step``
    (``attn.mla_kv_update_v2`` itself), starting from an empty cache — the
    task's option (ii), "run the HIR update token-by-token to fill the
    window": the pre-existing state a real serving loop would have arrived
    at, not a fabricated random cache (a random cache is reserved for the
    mechanism-only random-weight test, ``test_decode_step_e2e``, above)."""
    win = attn.REAL_WINDOW
    cache = torch.zeros(1, win, 1, attn.REAL_HEAD_DIM, dtype=torch.bfloat16, device=device)
    for pos in range(token_ids_seq.shape[0]):
        hidden = evaluate(decode_step.embed, weights["embed.table"], token_ids_seq[pos : pos + 1], device=device)
        cache = decode_step.kv_update_step(weights, hidden, pos, cache, device=device)
    return cache


@pytest.fixture(scope="module")
def layer0_real_weights() -> dict[str, torch.Tensor]:
    """Layer 0 fully real: attention (``mla_kv_update_v2`` / ``mla_attend_v2``
    real checkpoint weights) + hash-routed MoE (real checkpoint weights) +
    embed/norm/lm_head (real checkpoint weights). A *separate*
    ``WeightLoader`` instance from the ``loader``/``weights`` fixtures above
    — ``WeightLoader`` caches each post_init'd module's transformed weights
    forever after its first ``.load()`` call, so sharing the module-scoped
    ``loader`` here would silently keep serving the random-weight fixture's
    already-cached ``shared_expert`` dequant instead of dequanting these real
    weights."""
    raw = hf_weights.load_decode_step_weights(str(CKPT_DIR), layers=[0])
    raw.update(hf_weights.load_layer0_attention_weights(str(CKPT_DIR), device=DEVICE))
    return WeightLoader(decode_step.decode_step_module).load(raw)


@pytest.fixture(scope="module")
def learned_real_weights() -> dict[str, torch.Tensor]:
    """Real layer-3 learned-router (``moe_topk``) MoE + embed/norm/lm_head
    weights, retained to keep learned-router coverage now that the
    default/primary structure (``layer0_real_weights``, above) is
    hash-routed. Attention is real-**shape** random (see
    ``_random_attention_weights_v2``'s docstring: real layer 3's actual
    attention has ``compress_ratio==4`` Compressor/Indexer KV-compression
    sparsity, out of scope for ``attention.py``'s HIR port). Own
    ``WeightLoader`` instance, for the same reason as ``layer0_real_weights``."""
    raw = hf_weights.load_decode_step_weights(str(CKPT_DIR), layers=[LEARNED_LAYER])
    raw.update(_random_attention_weights_v2(torch.Generator(device=DEVICE).manual_seed(0)))
    return WeightLoader(decode_step.decode_step_module_learned).load(raw)


@pytest.mark.skipif(not CKPT_DIR.is_dir(), reason=f"real checkpoint not found at {CKPT_DIR}")
def test_decode_step_e2e_layer0_real_weights(layer0_real_weights, full_hash_leaves):
    """Layer 0 fully real, end to end: attention + hash-routed MoE +
    embed/norm/lm_head all real checkpoint weights. KV-cache pre-state via
    ``_prefill_kv_cache`` (T=200: past the REAL_WINDOW=128 boundary, so the
    "steady-state" wrapped-window regime). Gate: rel_l2<=1e-3,
    cosine>=0.9999 between the pure evaluator and the torch-cuda leaf path
    — attention is real this time, so the decoded token ids are not
    meaningless (unlike the retired GQA-placeholder version of this test),
    though only this one real transformer layer runs (not the full
    43-layer model), so the printed top-5 is a sanity signal, not a true
    next-token prediction."""
    device = DEVICE
    T = 200
    torch.manual_seed(0)
    prior_tokens = torch.randint(0, decode_step.VOCAB, (T,), device=device, dtype=torch.int64)
    cache = _prefill_kv_cache(layer0_real_weights, prior_tokens, device)
    token_ids = torch.randint(0, decode_step.VOCAB, (1,), device=device, dtype=torch.int64)

    def run(leaves):
        return decode_step.run_decode_step(
            layer0_real_weights, token_ids=token_ids, kv_cache_prev=cache, cur_pos=T,
            device=device, moe_kind="hash", leaves=leaves,
        )

    logits_ref, kv_ref = run(None)
    logits_leaf, kv_leaf = run(full_hash_leaves)

    assert logits_ref.shape == (1, 1, decode_step.VOCAB)
    assert torch.isfinite(logits_ref).all()
    assert torch.isfinite(logits_leaf).all()

    rel_l2 = _rel_l2(logits_leaf, logits_ref)
    cosine = _cosine(logits_leaf, logits_ref)
    top5 = torch.topk(logits_ref.reshape(-1).float(), k=5)
    print(
        f"\n[decode_step_e2e_layer0_real_weights T={T}] logits rel_l2={rel_l2:.4g} "
        f"cosine={cosine:.8f} (gate: rel_l2<=1e-3, cosine>=0.9999); "
        f"lm_head top-5 token ids={top5.indices.tolist()} logits={[round(v, 4) for v in top5.values.tolist()]}"
    )

    assert rel_l2 <= 1e-3
    assert cosine >= 0.9999
    torch.testing.assert_close(kv_leaf.float(), kv_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not CKPT_DIR.is_dir(), reason=f"real checkpoint not found at {CKPT_DIR}")
def test_decode_step_e2e_learned_moe_real_weights(learned_real_weights, full_learned_leaves):
    """Learned-router (``moe_topk``) coverage, kept alongside the (now
    default) hash-router layer-0 test above: real layer-3 MoE weights +
    embed/norm/lm_head, real-**shape** random attention (see
    ``learned_real_weights``'s docstring for why real layer-3 attention
    itself is out of scope). Same rel_l2/cosine gate and KV-cache-prestate
    convention (``_prefill_kv_cache``) as the layer0-real test above."""
    device = DEVICE
    T = 200
    torch.manual_seed(1)
    prior_tokens = torch.randint(0, decode_step.VOCAB, (T,), device=device, dtype=torch.int64)
    cache = _prefill_kv_cache(learned_real_weights, prior_tokens, device)
    token_ids = torch.randint(0, decode_step.VOCAB, (1,), device=device, dtype=torch.int64)

    def run(leaves):
        return decode_step.run_decode_step(
            learned_real_weights, token_ids=token_ids, kv_cache_prev=cache, cur_pos=T,
            device=device, moe_kind="learned", leaves=leaves,
        )

    logits_ref, kv_ref = run(None)
    logits_leaf, kv_leaf = run(full_learned_leaves)

    assert logits_ref.shape == (1, 1, decode_step.VOCAB)
    assert torch.isfinite(logits_ref).all()
    assert torch.isfinite(logits_leaf).all()

    rel_l2 = _rel_l2(logits_leaf, logits_ref)
    cosine = _cosine(logits_leaf, logits_ref)
    print(
        f"\n[decode_step_e2e_learned_moe_real_weights T={T}] logits rel_l2={rel_l2:.4g} "
        f"cosine={cosine:.8f} (gate: rel_l2<=1e-3, cosine>=0.9999; attention is real-shape random)"
    )

    assert rel_l2 <= 1e-3
    assert cosine >= 0.9999
    torch.testing.assert_close(kv_leaf.float(), kv_ref.float(), atol=ATOL, rtol=RTOL)


def test_post_init_ran_once_across_the_whole_module(loader):
    """Every test in this module that uses the ``weights`` fixture shares it
    (loaded once), so by the time this runs, shared_expert's post_init (M1)
    — the real fp8e4m3 (+ 128x128-block f32 scale) -> bf16 dequant — has run
    exactly once regardless of how many decode steps / cur_pos values / leaf
    registrations reused it. (The real-weight fixtures below use their own,
    separate ``WeightLoader`` instances — see their docstrings — so they do
    not affect this loader's count.)"""
    assert loader.post_init_runs == 1


def test_decode_step_leaf_diffs(weights, full_hash_leaves):
    """Per leaf: plain evaluator vs its registered torch-cuda implementation,
    each exercised in isolation (own inputs, own HIR Function) — the
    "逐 leaf 差分" complement to the end-to-end checks above. Covers every
    leaf in the default (hash-router) tree.

    ``moe_topk``'s equivalent isolated check (present before this fixture
    became hash-routed) is dropped — the ``weights`` fixture no longer
    carries a ``gate_bias`` (hash layers have none, see
    ``moe.moe_hash_gather``'s docstring), so there is no natural random data
    source for it here anymore. It is instead covered end to end by
    ``test_decode_step_e2e_learned_moe_real_weights`` above (pure evaluator
    vs its registered ``torch_moe_topk`` leaf, exercised through a real
    checkpoint layer) — see the report's assertion-removal list."""
    torch.manual_seed(999)
    hidden = (torch.randn(1, 1, DIM, device=DEVICE) * 0.02).to(torch.bfloat16)
    errors: dict[str, float] = {}

    def check(name, fn, args):
        ref = evaluate(fn, *args, device=DEVICE)
        got = full_hash_leaves[name].fn_or_source(*args)
        errors[name] = (ref.float() - got.float()).abs().max().item()
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=RTOL)

    table = (torch.randn(1000, DIM, device=DEVICE) * 0.02).to(torch.bfloat16)
    token_ids = torch.randint(0, 1000, (1,), device=DEVICE, dtype=torch.int64)
    check("embed", decode_step.embed, (table, token_ids))

    a = (torch.randn(1, 1, DIM, device=DEVICE) * 0.02).to(torch.bfloat16)
    b = (torch.randn(1, 1, DIM, device=DEVICE) * 0.02).to(torch.bfloat16)
    check("residual_add", decode_step.residual_add, (a, b))
    check("combine_expert_outputs", combine_expert_outputs, (a, b))

    check("final_rms_norm", decode_step.final_rms_norm, (hidden, weights["final_rms_norm.weight"]))
    check("lm_head", decode_step.lm_head, (hidden, weights["lm_head.weight"]))
    check("pre_moe_rms_norm", pre_moe_rms_norm, (hidden, weights["layer0.moe.rms_weight"]))

    check(
        "shared_expert", shared_expert,
        (
            hidden,
            weights["layer0.moe.shared_expert.w1_weight"], weights["layer0.moe.shared_expert.w1_scale"],
            weights["layer0.moe.shared_expert.w3_weight"], weights["layer0.moe.shared_expert.w3_scale"],
            weights["layer0.moe.shared_expert.w2_weight"], weights["layer0.moe.shared_expert.w2_scale"],
        ),
    )
    token_ids_hash = torch.randint(0, decode_step.VOCAB, (1,), device=DEVICE, dtype=torch.int64)
    check(
        "moe_hash_gather", moe_hash_gather,
        (
            hidden,
            weights["layer0.moe.gate_weight"], weights["layer0.moe.tid2eid"], token_ids_hash,
            weights["layer0.moe.routed_w1_weight"], weights["layer0.moe.routed_w1_scale"],
            weights["layer0.moe.routed_w3_weight"], weights["layer0.moe.routed_w3_scale"],
            weights["layer0.moe.routed_w2_weight"], weights["layer0.moe.routed_w2_scale"],
        ),
    )

    cur_pos_test = 7
    cos_pos, sin_pos = decode_step.rope_freqs_at(cur_pos_test, DEVICE)
    kv_cache_test = (
        torch.randn(1, attn.REAL_WINDOW, 1, attn.REAL_HEAD_DIM, device=DEVICE) * 0.02
    ).to(torch.bfloat16)
    cur_pos_t = torch.tensor([cur_pos_test % attn.REAL_WINDOW], dtype=torch.int32, device=DEVICE)
    s_one = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    check(
        "mla_kv_update_v2", attn.mla_kv_update_v2,
        (
            hidden, weights["layer0.attention.gamma_kv"], weights["layer0.attention.w_kv"],
            cos_pos, sin_pos, kv_cache_test, cur_pos_t, s_one,
        ),
    )

    ones_head_dim = torch.ones(attn.REAL_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    attn_mask = torch.zeros(1, 1, 1, attn.REAL_WINDOW, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.full((1, 1, 1, 1), attn.REAL_HEAD_DIM ** -0.5, device=DEVICE, dtype=torch.bfloat16)
    check(
        "mla_attend_v2", attn.mla_attend_v2,
        (
            hidden, weights["layer0.attention.gamma_q_lora"], weights["layer0.attention.w_q_a"],
            weights["layer0.attention.w_q_b"], ones_head_dim, cos_pos, sin_pos, kv_cache_test,
            attn_mask, weights["layer0.attention.attn_sink"], scale,
            weights["layer0.attention.w_o_a"], weights["layer0.attention.w_o_b"],
        ),
    )

    print("\n[decode_step_leaf_diffs] per-leaf max abs err:")
    for name, err in sorted(errors.items()):
        print(f"  {name:24s} {err:.4g}")
    assert set(errors) == set(full_hash_leaves)
