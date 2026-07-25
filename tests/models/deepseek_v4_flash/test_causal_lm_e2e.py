"""The one end-to-end test of the ``Module`` / ``RuntimeModule`` twin model,
on a real DeepSeek-V4-Flash subtree.

Deliberately a single test module with three tests rather than a spread of
narrow unit tests: everything below is on one path, so one run of it exercises
the parser (every ``@func`` in the tree is parsed at build time), ``@module``'s
three member kinds, weight derivation from ``ConstTensor`` params, per-weight
``.converter(...)`` registration and execution, ``RuntimeResource`` aliasing
(one-to-one, one-to-many, and segment renaming), ``prepare``'s offline
conversion and strict shape/dtype validation, both sides' ``load``,
``@runtime_module``'s decoration-time structure check, weight-by-name filling
at call time, orchestration reuse, and the shared ``generate`` loop.

    raw checkpoint (real DeepSeek key names, fabricated)
      │  SafetensorsResource(alias=hf_alias(config))     ← ckpt naming lives only here
      ▼
    Module.prepare  ─ per-weight converters + one-to-many stacking ─►  prepared dir
      │                                                                 (clean module paths)
      ├─► Module.load          → evaluator kernels     = reference
      └─► RuntimeModule.load   → torch / CUDA kernels  = candidate
                                        │
                          generate(m, ids, 2) on both → parity

Scale: ``DSV4Config.tiny()``. Not a toy architecture — the same code builds the
real 43-layer model from the checked-in ``config.json``; only the sizes shrink,
and they stay 128-divisible so the fp8 block-scale grid is a real 2x2 rather
than degenerating to 1x1. The whole fabricated checkpoint is a few hundred KB.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.models.deepseek_v4_flash.causal_lm import build_causal_lm
from tests.models.deepseek_v4_flash.config import DSV4Config
from tests.models.deepseek_v4_flash.hf_alias import hf_alias
from tests.models.deepseek_v4_flash.runtime import build_runtime_causal_lm
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.runtime import (
    SafetensorsResource,
    bench,
    check,
    generate,
    runtime_func,
    runtime_module,
)
from tilefoundry.target.cuda import H200SXM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Every tensor the prepared checkpoint should contain, spelled out: a clean
# module path per declared weight, with none of the checkpoint's own vocabulary
# (`layers.0`, `attn`, `ffn`, `experts.3`, `wkv`, ...) surviving into it.
EXPECTED_PREPARED_KEYS = {
    "table",
    "final_norm_weight",
    "lm_head_weight",
    "layer0.pre_attn_norm_weight",
    "layer0.pre_moe_norm_weight",
    "layer0.attention.gamma_kv",
    "layer0.attention.gamma_q_lora",
    "layer0.attention.w_kv",
    "layer0.attention.w_q_a",
    "layer0.attention.w_q_b",
    "layer0.attention.attn_sink",
    "layer0.attention.w_o_a",
    "layer0.attention.w_o_b",
    "layer0.moe.gate_weight",
    "layer0.moe.tid2eid",
    "layer0.moe.w1_weight",
    "layer0.moe.w1_scale",
    "layer0.moe.w3_weight",
    "layer0.moe.w3_scale",
    "layer0.moe.w2_weight",
    "layer0.moe.w2_scale",
    "layer0.moe.shared_w1_weight",
    "layer0.moe.shared_w1_scale",
    "layer0.moe.shared_w3_weight",
    "layer0.moe.shared_w3_scale",
    "layer0.moe.shared_w2_weight",
    "layer0.moe.shared_w2_scale",
}


def _fabricate_one(shape, dtype, name, generator):
    """One raw checkpoint tensor. Block scales get *non-unit* powers of two so
    that dequantization is a real transformation: with all-ones scales a
    dequantizing converter would be indistinguishable from a plain cast, and
    the parity assertions below would pass on an identity ``prepare``."""
    dt = to_torch_dtype(dtype)
    if dt in (torch.int32, torch.int64):
        return torch.zeros(shape, dtype=dt)  # tid2eid: expert 0 is always valid
    if dtype.name in ("f8e8m0", "f32") and "scale" in name:
        exponents = torch.randint(-2, 3, shape, generator=generator, dtype=torch.int64)
        return torch.pow(2.0, exponents.to(torch.float32)).to(dt)
    values = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
    return values.to(dt)


def _raw_key(path, leaf, alias):
    """The key ``RuntimeResource`` will look up for *leaf* at tree *path*: every
    path segment and the leaf mapped through *alias*, joined onto the
    accumulated prefix — ``resource.py``'s ``_resolve_segment`` /
    ``_resolve_key`` (this table carries bare entries only, so a plain ``get``
    matches its path-qualified-then-bare order)."""
    prefix = "".join(f"{alias.get(seg, seg)}." for seg in path)
    hit = alias.get(leaf, leaf)
    if isinstance(hit, tuple):
        return tuple(f"{prefix}{one}" for one in hit)
    return f"{prefix}{hit}"


def _fabricate_checkpoint(mod, alias, generator, path=(), out=None):
    """The raw checkpoint the tree needs, derived from the tree's own
    declarations rather than hand-listed: a weight with a converter needs that
    converter's raw params (raw names, raw shapes/dtypes), a weight without one
    is its own raw form, and a tuple-valued alias means N per-shard tensors of
    the declared shape minus the leading axis ``prepare`` stacks along."""
    out = {} if out is None else out
    converters = {w: c for fn in mod.functions for w, c in getattr(fn, "converters", ())}
    for weight, declared in mod.weights.items():
        conv = converters.get(weight)
        needed = [(p.name, p.type) for p in conv.params] if conv else [(weight, declared)]
        for name, ty in needed:
            key = _raw_key(path, name, alias)
            shape = tuple(int(d) for d in ty.shape)
            if isinstance(key, tuple):
                for one in key:
                    out[one] = _fabricate_one(shape[1:], ty.dtype, name, generator)
            else:
                out[key] = _fabricate_one(shape, ty.dtype, name, generator)
    for child in mod.modules:
        _fabricate_checkpoint(child, alias, generator, (*path, child.name), out)
    return out


def _write_checkpoint_dir(tensors, out_dir, shards=2):
    """Write *tensors* as a repacked HF-style checkpoint: N safetensors shards
    plus ``model.safetensors.index.json``. More than one shard on purpose —
    ``SafetensorsResource`` resolves each name through the index to its own
    shard, so a single-shard checkpoint would not exercise that."""
    names = sorted(tensors)
    weight_map = {}
    for i in range(shards):
        part = names[i::shards]
        shard = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: tensors[n].contiguous() for n in part}, str(out_dir / shard))
        weight_map.update({n: shard for n in part})
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
             "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)


def _dequant_blocks(weight, scale):
    """Reference fp8 block dequantization: each ``scale[i, j]`` covers one
    tile of *weight*, sized by the two shapes' ratio."""
    rows, cols = weight.shape
    bo, bi = scale.shape
    tiled = weight.to(torch.float32).reshape(bo, rows // bo, bi, cols // bi)
    return (tiled * scale.to(torch.float32)[:, None, :, None]).reshape(rows, cols)


@pytest.fixture(scope="module")
def config():
    return DSV4Config.tiny()


@pytest.fixture(scope="module")
def semantic(config):
    """One tree, built once: building it parses every `@func` in the model, and
    fabricating / preparing / loading all read the same declarations."""
    return build_causal_lm(config)


@pytest.fixture(scope="module")
def raw_tensors(config, semantic):
    """The fabricated raw checkpoint, keyed by real DeepSeek-V4 names."""
    generator = torch.Generator().manual_seed(20260725)
    return _fabricate_checkpoint(semantic, hf_alias(config), generator)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, config, semantic, raw_tensors):
    """``prepare`` once, from a real on-disk raw checkpoint read through an
    aliasing ``SafetensorsResource``."""
    raw_dir = tmp_path_factory.mktemp("dsv4_raw")
    _write_checkpoint_dir(raw_tensors, raw_dir)
    out_dir = tmp_path_factory.mktemp("dsv4_prepared")
    raw = SafetensorsResource(str(raw_dir), device="cpu", alias=hf_alias(config))
    semantic.prepare(raw, str(out_dir), device="cpu")
    return out_dir


@pytest.fixture(scope="module")
def twins(config, semantic, prepared):
    """The two loaded trees under test: the semantic module (evaluator kernels,
    the reference) and its runtime twin (torch / CUDA kernels, the candidate),
    both loaded from the same prepared checkpoint."""
    runtime = build_runtime_causal_lm(config, ir=semantic)
    semantic.load(SafetensorsResource(str(prepared), device="cuda"))
    runtime.load(SafetensorsResource(str(prepared), device="cuda"))
    return semantic, runtime


def _node_inputs(semantic, config):
    """One activation tuple per node, taken from the model's own generation
    hooks rather than fabricated: whatever ``prepare_inputs_for_generation``
    hands ``forward`` at decode step 0 is by construction the real thing."""
    ids = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    caches = semantic.init_caches(device="cuda")
    root_args = semantic.prepare_inputs_for_generation(ids, 0, caches, device="cuda")
    token_ids, cos, sin, cur_pos, s, past, mask, scale, ones = root_args
    hidden = semantic.embed(token_ids)
    attention_args = (hidden, cos, sin, cur_pos, s, past[0], mask, scale, ones)
    return {
        "attention": attention_args,
        "moe": (hidden, token_ids),
        "layer0": (*attention_args, token_ids),
        "root": root_args,
    }


def test_prepare_and_parity(config, raw_tensors, prepared, twins):
    """``prepare`` digests the checkpoint's naming and really converts, and the
    twin agrees with the evaluator node by node."""
    semantic, runtime = twins
    store = SafetensorsResource(str(prepared), device="cuda")

    # ── the checkpoint's vocabulary does not survive prepare ───────────────
    assert set(store._index()) == EXPECTED_PREPARED_KEYS
    debris = [k for k in store._index() if any(
        tag in k for tag in ("layers.", "attn.", "ffn", "experts.", "wkv", "wq_", "wo_", "raw")
    )]
    assert debris == []

    # ── a converter with two raw inputs really ran: offline block dequant of
    #    an fp8 weight, then the transpose the HIR's matmul convention needs ──
    w_kv = store.load("layer0.attention.w_kv")
    expected_w_kv = _dequant_blocks(
        raw_tensors["layers.0.attn.wkv.weight"], raw_tensors["layers.0.attn.wkv.scale"]
    ).t().to(torch.bfloat16)
    assert w_kv.shape == expected_w_kv.shape
    torch.testing.assert_close(w_kv.float(), expected_w_kv.float().cuda(), rtol=0, atol=0)
    # ... and is not the identity: the raw tensor is fp8, this is dequantized bf16
    assert raw_tensors["layers.0.attn.wkv.weight"].dtype == torch.float8_e4m3fn
    assert w_kv.dtype == torch.bfloat16
    assert (raw_tensors["layers.0.attn.wkv.scale"].float() != 1.0).any(), "scales must be non-unit"

    # ── one-to-many alias, pass-through: N per-expert tensors stacked, in
    #    declared order, byte for byte ────────────────────────────────────────
    w1 = store.load("layer0.moe.w1_weight")
    expected_w1 = torch.stack([
        raw_tensors[f"layers.0.ffn.experts.{i}.w1.weight"] for i in range(config.n_routed)
    ])
    assert w1.shape == expected_w1.shape
    assert torch.equal(w1.float().cpu(), expected_w1.float())

    # ── one-to-many alias, then a converter: stacked f32 shards, cast to the
    #    declared exponent-only scale dtype ─────────────────────────────────
    w1_scale = store.load("layer0.moe.w1_scale")
    expected_scale = torch.stack([
        raw_tensors[f"layers.0.ffn.experts.{i}.w1.scale"] for i in range(config.n_routed)
    ])
    assert raw_tensors["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float32
    assert w1_scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(w1_scale.float().cpu(), expected_scale)

    # ── the twin agrees with the evaluator, innermost node outward ──────────
    inputs = _node_inputs(semantic, config)
    nodes = {
        "attention": (runtime.layer0.attention, semantic.layer0.attention),
        "moe": (runtime.layer0.moe, semantic.layer0.moe),
        "layer0": (runtime.layer0, semantic.layer0),
        "root": (runtime, semantic),
    }
    for name, (candidate, reference) in nodes.items():
        report = check(candidate.forward, reference.forward, inputs[name])
        assert report.passed, f"{name}: {dict(report.metrics)}"
        # every node reuses the semantic module's orchestration: none of the
        # runtime classes authors a forward of its own.
        assert "forward" not in vars(type(candidate)), name


def test_generate_two_steps(config, twins):
    """Two decode steps through the shared ``generate`` loop, with state held
    entirely by the caller and threaded functionally in and out."""
    semantic, runtime = twins
    ids = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")

    reference_logits, reference_caches = generate(semantic, ids, 2, device="cuda")
    candidate_logits, candidate_caches = generate(runtime, ids, 2, device="cuda")

    assert len(candidate_logits) == len(reference_logits) == 2
    report = check(
        lambda: (candidate_logits, candidate_caches),
        lambda: (reference_logits, reference_caches),
        (),
    )
    assert report.passed, dict(report.metrics)

    # the cache is real state: two steps wrote exactly two window slots ...
    occupied = (reference_caches[0].float().abs().sum(dim=(0, 2, 3)) > 0)
    assert int(occupied.sum()) == 2
    assert bool(occupied[0]) and bool(occupied[1])
    # ... and every logit is finite (the hand-written YaRN rope, the fp8
    # dequant path and the fused MoE all stay in range)
    for step, logits in enumerate(reference_logits):
        assert torch.isfinite(logits.float()).all(), step

    # ... held by the caller, never mutated in place: forward returns a new
    # cache and leaves the one it was given untouched, on both sides.
    for model in (semantic, runtime):
        caches = model.init_caches(device="cuda")
        args = model.prepare_inputs_for_generation(ids, 0, caches, device="cuda")
        _, next_caches = model(*args)
        assert next_caches[0] is not caches[0]
        assert not caches[0].float().any(), type(model).__name__
        assert next_caches[0].float().any()


def test_structure_mismatch_rejected(config, twins):
    """``@runtime_module`` requires a strict one-to-one structure at decoration
    time — the twin is the same tree with kernel leaves swapped in, so a
    missing or extra kernel is an authoring error, not a runtime surprise."""
    semantic, runtime = twins
    attention = semantic.layer0.attention
    layer = semantic.layer0

    with pytest.raises(TypeError, match=r"missing \['mla_kv_update'\]"):
        @runtime_module(attention)
        class MissingKernel:
            @runtime_func
            def mla_attend(self, *args):
                raise AssertionError("never called")

    with pytest.raises(TypeError, match=r"extra \['mla_attend_extra'\]"):
        @runtime_module(attention)
        class ExtraKernel:
            @runtime_func
            def mla_kv_update(self, *args):
                raise AssertionError("never called")

            @runtime_func
            def mla_attend(self, *args):
                raise AssertionError("never called")

            @runtime_func
            def mla_attend_extra(self, *args):
                raise AssertionError("never called")

    with pytest.raises(TypeError, match=r"child module names.*missing \['moe'\]"):
        @runtime_module(layer)
        class MissingChild:
            attention = type(runtime.layer0.attention)

            @runtime_func
            def pre_attn_rms_norm(self, *args):
                raise AssertionError("never called")

            @runtime_func
            def pre_moe_rms_norm(self, *args):
                raise AssertionError("never called")

            @runtime_func
            def residual_add(self, *args):
                raise AssertionError("never called")

    # `bench` runs over the same twin the parity check used
    inputs = _node_inputs(semantic, config)["attention"]
    report = bench(runtime.layer0.attention.forward, inputs, iters=3, device=H200SXM())
    assert report.metrics["mean_ms"] > 0
    assert report.passed is None
