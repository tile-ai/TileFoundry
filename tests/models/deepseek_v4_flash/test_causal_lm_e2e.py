"""End-to-end test of the ``Module`` / ``RuntimeModule`` twin model on a real
DeepSeek-V4-Flash subtree: prepare a fabricated checkpoint, load both sides,
check parity.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.models.deepseek_v4_flash.config import TINY
from tests.models.deepseek_v4_flash.hf_alias import hf_alias
from tests.models.deepseek_v4_flash.model import build_deepseek_v4_flash
from tests.models.deepseek_v4_flash.runtime import build_runtime_causal_lm
from tests.models.generation import generate
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.ir.core.module import Module
from tilefoundry.runtime import (
    SafetensorsResource,
    bench,
    check,
    runtime_func,
    runtime_module,
)
from tilefoundry.target.cuda import CudaTarget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

#: How far the twin follows the loop. Two steps is what it takes for one step to
#: read the previous one's output; the remaining steps only fill the window, and
#: running the runtime side over all of them buys the same fact per step.
TWINNED_STEPS = 2

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
    dt = to_torch_dtype(dtype)
    if dt in (torch.int32, torch.int64):
        return torch.zeros(shape, dtype=dt)
    if dtype.name in ("f8e8m0", "f32") and "scale" in name:
        exponents = torch.randint(-2, 3, shape, generator=generator, dtype=torch.int64)
        return torch.pow(2.0, exponents.to(torch.float32)).to(dt)
    values = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
    return values.to(dt)


def _raw_key(path, leaf, alias):
    prefix = "".join(f"{alias.get(seg, seg)}." for seg in path)
    hit = alias.get(leaf, leaf)
    if isinstance(hit, tuple):
        return tuple(f"{prefix}{one}" for one in hit)
    return f"{prefix}{hit}"


def _fabricate_checkpoint(mod, alias, generator, path=(), out=None):
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
    rows, cols = weight.shape
    bo, bi = scale.shape
    tiled = weight.to(torch.float32).reshape(bo, rows // bo, bi, cols // bi)
    return (tiled * scale.to(torch.float32)[:, None, :, None]).reshape(rows, cols)


@pytest.fixture(scope="module")
def config():
    return TINY


@pytest.fixture(scope="module")
def semantic(config):
    """The whole tree at the small shape: the size this end-to-end path is
    affordable at, named rather than derived."""
    return build_deepseek_v4_flash(config)


@pytest.fixture(scope="module")
def raw_tensors(config, semantic):
    generator = torch.Generator().manual_seed(20260725)
    return _fabricate_checkpoint(semantic, hf_alias(config), generator)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, config, semantic, raw_tensors):
    raw_dir = tmp_path_factory.mktemp("dsv4_raw")
    _write_checkpoint_dir(raw_tensors, raw_dir)
    out_dir = tmp_path_factory.mktemp("dsv4_prepared")
    raw = SafetensorsResource(str(raw_dir), device="cpu", alias=hf_alias(config))
    semantic.prepare(raw, str(out_dir), device="cpu")
    return out_dir


@pytest.fixture(scope="module")
def twins(config, semantic, prepared):
    runtime = build_runtime_causal_lm(config, ir=semantic)
    # The ir module's `load` returns; the runtime twin's still binds in place.
    loaded = semantic.load(SafetensorsResource(str(prepared), device="cuda"))
    runtime.load(SafetensorsResource(str(prepared), device="cuda"))
    return loaded, runtime


def _node_inputs(semantic, config):
    ids = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    caches = semantic.init_caches(device="cuda")
    root_args = semantic.prepare_inputs_for_generation(ids, 0, caches, device="cuda")
    token_ids, cos, sin, past, scale, ones = root_args
    hidden = semantic.embed(token_ids)
    attention_args = (hidden, cos, sin, past[0], scale, ones)
    return {
        "attention": attention_args,
        "moe": (hidden, token_ids),
        "layer0": (*attention_args, token_ids),
        "root": root_args,
    }


def test_prepare_and_parity(config, raw_tensors, prepared, twins):
    """``prepare`` digests the checkpoint's naming and really converts, one leaf of
    the prepared store loads on its own, and the twin agrees with the evaluator node
    by node."""
    semantic, runtime = twins
    store = SafetensorsResource(str(prepared), device="cuda")

    assert set(store._index()) == EXPECTED_PREPARED_KEYS
    debris = [k for k in store._index() if any(
        tag in k for tag in ("layers.", "attn.", "ffn", "experts.", "wkv", "wq_", "wo_", "raw")
    )]
    assert debris == []

    w_kv = store.load("layer0.attention.w_kv")
    expected_w_kv = _dequant_blocks(
        raw_tensors["layers.0.attn.wkv.weight"], raw_tensors["layers.0.attn.wkv.scale"]
    ).t().to(torch.bfloat16)
    assert w_kv.shape == expected_w_kv.shape
    torch.testing.assert_close(w_kv.float(), expected_w_kv.float().cuda(), rtol=0, atol=0)
    assert raw_tensors["layers.0.attn.wkv.weight"].dtype == torch.float8_e4m3fn
    assert w_kv.dtype == torch.bfloat16
    assert (raw_tensors["layers.0.attn.wkv.scale"].float() != 1.0).any(), "scales must be non-unit"

    w1 = store.load("layer0.moe.w1_weight")
    expected_w1 = torch.stack([
        raw_tensors[f"layers.0.ffn.experts.{i}.w1.weight"] for i in range(config.n_routed)
    ])
    assert w1.shape == expected_w1.shape
    assert torch.equal(w1.float().cpu(), expected_w1.float())

    w1_scale = store.load("layer0.moe.w1_scale")
    expected_scale = torch.stack([
        raw_tensors[f"layers.0.ffn.experts.{i}.w1.scale"] for i in range(config.n_routed)
    ])
    assert raw_tensors["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float32
    assert w1_scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(w1_scale.float().cpu(), expected_scale)

    # One leaf against its own slice of the same store, with `prepare` patched to
    # raise so that reaching it would fail here.
    leaf = semantic.layer0.moe.module
    with pytest.MonkeyPatch.context() as patched:
        def refuse(*_args, **_kwargs):
            raise AssertionError("loading one leaf reached Module.prepare")

        patched.setattr(Module, "prepare", refuse)
        loaded_leaf = leaf.load(store.subtree("layer0").subtree("moe"))

    assert loaded_leaf.module is leaf
    assert set(loaded_leaf.constants) == set(leaf.weights)
    assert {f"layer0.moe.{name}" for name in loaded_leaf.constants} == {
        key for key in EXPECTED_PREPARED_KEYS if key.startswith("layer0.moe.")
    }
    assert torch.equal(
        loaded_leaf.constants["w1_weight"].float(), store.load("layer0.moe.w1_weight").float()
    )

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
        assert "forward" not in vars(type(candidate)), name


def test_the_generate_loop_threads_state_and_stops_at_the_window(config, twins):
    """The decode loop over enough steps to fill the window, from three angles a
    single run answers at once.

    State is held entirely by the caller and threaded functionally in and out, so
    the cache each step reads is the context before its token: what a step hands
    back is that token's own position, and the caller appending it is what makes
    the second step read the first one's output. Past the window the caller stops
    growing it -- eviction is a policy applied to a tensor, which is why the
    description carries a range and not a fixed capacity, and why it is visible
    here rather than buried in a write index. And the twin runs the same loop:
    checked over the steps both sides take, so a runtime kernel that broke the
    threading would disagree with the evaluator rather than merely look plausible.
    """
    semantic, runtime = twins
    steps = config.window + 2
    ids = torch.arange(steps, dtype=torch.int64, device="cuda") % config.vocab
    seed_ctx = semantic.init_caches(device="cuda")[0].shape[1]

    reference_logits, reference_caches = generate(semantic, ids, steps, device="cuda")
    candidate_logits, candidate_caches = generate(runtime, ids, TWINNED_STEPS, device="cuda")

    assert len(reference_logits) == steps
    assert len(candidate_logits) == TWINNED_STEPS
    report = check(
        lambda: candidate_logits,
        lambda: reference_logits[:TWINNED_STEPS],
        (),
    )
    assert report.passed, dict(report.metrics)
    for step, logits in enumerate(reference_logits):
        assert torch.isfinite(logits.float()).all(), step

    # The window is a capacity: the context stops at what the layer can attend.
    assert reference_caches[0].shape[1] == config.max_ctx
    assert config.max_ctx == config.window - 1
    assert candidate_caches[0].shape[1] == seed_ctx + TWINNED_STEPS

    for model in (semantic, runtime):
        caches = model.init_caches(device="cuda")
        args = model.prepare_inputs_for_generation(ids, 0, caches, device="cuda")
        _, fresh = model(*args)
        # A step hands back one position per layer -- this token's own latent --
        # and appending it leaves the context it was given unchanged with that
        # position after it.
        assert fresh[0].shape[1] == 1, type(model).__name__
        next_caches = model.append_cache(caches, fresh)
        assert next_caches[0].shape[1] == caches[0].shape[1] + 1, type(model).__name__
        assert torch.equal(next_caches[0][:, :seed_ctx], caches[0])
        assert next_caches[0][:, seed_ctx:].float().any()


def test_structure_mismatch_rejected(config, twins):
    """``@runtime_module`` rejects a missing, extra, or mismatched kernel/child
    at decoration time."""
    semantic, runtime = twins
    # `runtime_module` decorates against the IR's function and child names, so it
    # takes the Module rather than this loading of it.
    attention = semantic.layer0.attention.module
    layer = semantic.layer0.module

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

    inputs = _node_inputs(semantic, config)["attention"]
    report = bench(runtime.layer0.attention.forward, inputs, iters=3, device=CudaTarget().device)
    assert report.metrics["mean_ms"] > 0
    assert report.passed is None
