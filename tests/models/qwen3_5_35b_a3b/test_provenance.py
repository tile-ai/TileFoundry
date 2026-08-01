"""What this fixture is sourced from, and what it does not cover.

A model package named after a checkpoint makes two claims a reader cannot check
by reading it: that its numbers came from that checkpoint, and that the tests
next to it cover what they appear to. Both are asserted here.

The coverage half is the more important one for a model this size. The published
model is 40 layers and 35 billion parameters, and the reference for it is
declared, boundary-complete submodules rather than a stack -- so what is *not*
executed has to be written down, or "the tests pass" would read as a claim about
the model. ``NOT_EXECUTED`` is that list, and it is checked against the Modules
themselves: a function nobody runs has to be declared here before the suite will
accept it.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from tests.models.qwen3_5_35b_a3b import reference
from tests.models.qwen3_5_35b_a3b.model import (
    _ROPE_ROWS,
    _ROT,
    LAYER_TYPE,
    Qwen3_5_35B_A3B,
    Qwen3_5FullAttention,
    Qwen3_5FullAttentionDecoderLayer,
    Qwen3_5LinearAttention,
    Qwen3_5LinearAttentionDecoderLayer,
    Qwen3_5MoE,
    Qwen3_5Router,
)
from tilefoundry.ir.core.module import select

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# Where the numbers came from is `tests/models/test_provenance.py`: it pins this
# fixture's `config.json` to the revision it was downloaded at, over the whole
# published file rather than over a copied subset of its fields. What is left
# here is what that cannot say -- which parts of the published model this fixture
# covers, and which it does not.


def test_the_stack_is_the_published_order_and_its_layers_are_independent():
    """The stack is `layer_types` in order, and each layer is an independent copy.

    A stack built from a restated pattern, or from layers sharing Functions, would
    pass every component test in this package and fail here.
    """
    shape = reference.CONFIG
    stack = Qwen3_5_35B_A3B
    assert len(stack.modules) == shape.num_hidden_layers
    assert [child.name for child in stack.modules[:3]] == ["layer0", "layer1", "layer2"]

    for index, kind in enumerate(shape.layer_types):
        mixer = select(stack, f"layer{index}.mixer")
        entry = LAYER_TYPE[kind].modules[0].entry
        assert mixer.entry == entry, f"layer{index} is not a {kind} layer"

    first, second = stack.modules[0], stack.modules[1]
    assert first.lookup("residual_add") is not second.lookup("residual_add")
    assert first.modules[0].lookup("conv_step") is not second.modules[0].lookup("conv_step")

    selected = select(stack, "layer0.moe.experts")
    assert selected.name == "moe"
    assert selected.entry_function().name == "experts"
    # The router is a Module of its own, two levels down, and reachable as one.
    assert select(stack, "layer0.moe.router.routing").entry_function().name == "routing"

    assert "forward" in stack.methods
    with pytest.raises(ValueError, match=f"{shape.num_hidden_layers} layers but was given 0"):
        stack.decode_hidden(None, (), ())


# ── what mrope actually covers here ─────────────────────────────────────


def test_mrope_degenerates_in_a_text_only_fixture():
    """Measured, not assumed: with one position per token, mrope is plain RoPE.

    The published rotary embedding assigns each token a position *triple* and
    interleaves the three axes' frequencies by ``mrope_section`` ``[11, 11, 10]``.
    A text-only fixture gives all three axes the same number, so every branch of
    the interleave selects the same frequency. This measures that the published
    module's output equals an ordinary partial RoPE at ``rotary_dim``, which is
    what these tests therefore cover -- the partial factor, not mrope.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeTextRotaryEmbedding,
    )

    shape = reference.CONFIG
    partial_rotary_factor = float(shape.rope_parameters["partial_rotary_factor"])
    rotary_dim = int(shape.head_dim * partial_rotary_factor)
    total = 64
    cos, sin = reference.rope_caches_at(total=total, device=DEV, dtype=torch.float32)
    assert cos.shape == (total, rotary_dim), (
        "the caches are rotary_dim wide, not head_dim: partial_rotary_factor "
        f"{partial_rotary_factor} of {shape.head_dim}"
    )

    with torch.device(DEV):
        rotary = Qwen3_5MoeTextRotaryEmbedding(reference.CONFIG)
    frequencies = torch.outer(
        torch.arange(total, device=DEV).float(), rotary.inv_freq
    )
    plain = torch.cat([frequencies, frequencies], dim=-1)

    assert (plain.cos() - cos).abs().max().item() < 1e-9
    assert (plain.sin() - sin).abs().max().item() < 1e-9


def test_generation_inputs_match_the_drawn_position():
    """The root supplies positional activations to full-attention layers only."""
    step = 24
    source = torch.arange(step + 2, dtype=torch.int64, device=DEV)
    sentinel = object()

    token_ids, layer_args, caches = Qwen3_5_35B_A3B.prepare_inputs_for_generation(
        source[: step + 1], step, sentinel, device=DEV
    )
    want_cos, want_sin = reference.rope_caches_at(total=step + 1, device=DEV)

    assert torch.equal(token_ids, source[step].reshape(1))
    assert len(layer_args) == len(reference.CONFIG.layer_types)
    for kind, args in zip(reference.CONFIG.layer_types, layer_args):
        if kind == "linear_attention":
            assert args == ()
            continue
        cos, sin, pos_ids, scale = args
        assert torch.equal(cos[: step + 1], want_cos)
        assert torch.equal(sin[: step + 1], want_sin)
        assert cos.shape == sin.shape == (_ROPE_ROWS, _ROT)
        assert torch.equal(pos_ids, torch.tensor([step], device=DEV, dtype=torch.int32))
        assert scale.shape == (1, 1, 1, 1)
    assert caches is sentinel


# ── the multi-token-prediction gap ──────────────────────────────────────


def test_multi_token_prediction_has_no_oracle_in_the_installed_transformers():
    """``mtp_num_hidden_layers`` is 1, and there is nothing to compare against.

    This is a gate on a limit that was measured, not expected. The published
    configuration states a multi-token-prediction head; the installed
    transformers implements none -- ``mtp`` appears in the Qwen3.5 modeling file
    only in ``_keys_to_ignore_on_load_unexpected``, which *discards* those
    weights on load, and no class, function or config field in the package
    defines the head's semantics.

    Without an oracle, a reference for it could only be this repository's own
    reading of the configuration, compared against this repository's own kernels.
    That proves nothing, so it is not written. If a future transformers ships the
    head, this test fails and the gate has to be lifted deliberately.
    """
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling  # noqa: PLC0415

    assert reference.CONFIG.mtp_num_hidden_layers == 1

    source = Path(inspect.getfile(modeling)).read_text()
    mentions = [line.strip() for line in source.splitlines() if "mtp" in line]
    assert mentions, "expected the discard rule to be present"
    assert all("_keys_to_ignore_on_load_unexpected" in line for line in mentions), (
        f"transformers now mentions mtp outside its discard rule: {mentions}"
    )
    # Nothing in the module namespace defines the head either, under any spelling.
    assert not [
        name for name, _ in inspect.getmembers(modeling)
        if "mtp" in name.lower()
    ]


# ── declared coverage ───────────────────────────────────────────────────

#: Every IR function this package authors, and whether a test runs it. A function
#: that is neither executed nor declared unexecuted fails
#: ``test_every_authored_function_is_either_executed_or_declared``, so coverage
#: cannot drift by someone adding a kernel.
EXECUTED: dict[str, tuple[str, ...]] = {
    "Qwen3_5Router": ("routing",),
    "Qwen3_5MoE": ("post_norm", "routed_experts", "shared_expert", "experts"),
    "Qwen3_5FullAttention": ("partial_rope", "full_attention"),
    "Qwen3_5LinearAttention": ("linear_attention",),
    "Qwen3_5FullAttentionDecoderLayer": ("residual_add",),
    "Qwen3_5LinearAttentionDecoderLayer": ("residual_add",),
}

NOT_EXECUTED: dict[str, tuple[str, ...]] = {
    # Same body as `partial_rope` over the key head count, exercised through
    # `full_attention`, never called on its own.
    "Qwen3_5FullAttention": ("partial_rope_kv",),
    # Exercised through `linear_attention`. Not called directly because neither
    # has a Hugging Face module of its own to compare against: `conv_step` is a
    # fragment of `Qwen3_5MoeGatedDeltaNet.forward` and `delta_step` a single
    # token of `torch_recurrent_gated_delta_rule`, so a direct comparison would
    # be against a slice of a function rather than against a module boundary.
    "Qwen3_5LinearAttention": ("conv_step", "l2_normalise", "delta_step"),
    # The stack these belong to is not run against an oracle (see
    # `the_40_layer_stack` below), so neither the norm that closes it nor the
    # embedding and head at its two ends have anything to be compared with. They
    # are authored because the model has them, and without them the root would
    # describe a step that begins and ends nowhere.
    "Qwen3_5_35B_A3B": ("embed", "final_rms_norm", "lm_head"),
}

#: What no test in this package executes at all, with the reason. Not derived
#: from the code -- it is the part a reader cannot see by looking at what is
#: there, so it is stated.
UNCOVERED_SEMANTICS: dict[str, str] = {
    "multi_token_prediction": (
        "mtp_num_hidden_layers is 1 and the installed transformers implements no "
        "head for it (see the test above). No oracle, so no reference."
    ),
    "vision_tower": (
        "the published model is multimodal; only the text tower is in scope, so "
        "vision_config and every Qwen3_5MoeVision* module are untouched."
    ),
    "the_40_layer_stack": (
        "`Qwen3_5_35B_A3B` composes the published order and orchestrates it, so the "
        "stack is described, its layer order is checked and its step is reachable "
        "-- but no complete-stack reference is built for a model this size, so the "
        "residual thread across 40 layers and the final norm are never compared "
        "with anything; one layer of each published type is."
    ),
    "embedding_and_lm_head": (
        "the only vocabulary-sized weights in the model. Both are authored on the "
        "root now and declared as weights it holds, and neither is compared with "
        "anything: the oracle would be the whole 40-layer text tower, and 35 "
        "billion parameters in f32 do not fit one card. The head's converter -- "
        "the transpose of the published (vocab, hidden) layout -- is authored for "
        "`prepare`, which nothing here calls either."
    ),
    "mrope": (
        "a text-only fixture gives all three position axes the same value, so "
        "the mrope_section interleave is the identity (measured above). What is "
        "covered is partial_rotary_factor, not mrope."
    ),
    "prefill_and_chunked_decode": (
        "seq_len is the literal 1 everywhere here. The chunked delta rule and "
        "the multi-position attention path are used only to build the oracle's "
        "state, never as kernels under test."
    ),
    "bf16": (
        "every kernel is authored and compared at f32. The published dtype is "
        "bfloat16; what a bf16 comparison would and would not resolve is not "
        "measured by this package."
    ),
    "quantisation_and_tensor_parallelism": (
        "the published config carries tp/ep plans; nothing here shards or "
        "quantises anything."
    ),
}


def _authored_modules():
    return (
        Qwen3_5Router,
        Qwen3_5MoE,
        Qwen3_5FullAttention,
        Qwen3_5LinearAttention,
        Qwen3_5FullAttentionDecoderLayer,
        Qwen3_5LinearAttentionDecoderLayer,
        Qwen3_5_35B_A3B,
    )


def test_every_authored_function_is_either_executed_or_declared():
    """No kernel joins this package without a test or an entry saying why not."""
    for module in _authored_modules():
        authored = {function.name for function in module.functions}
        executed = set(EXECUTED.get(module.name, ()))
        declared = set(NOT_EXECUTED.get(module.name, ()))

        assert not (executed & declared), (module.name, executed & declared)
        assert executed | declared == authored, (
            f"{module.name}: authored {sorted(authored)}, but EXECUTED names "
            f"{sorted(executed)} and NOT_EXECUTED names {sorted(declared)}"
        )


def test_the_uncovered_semantics_are_stated_with_reasons():
    """A gap with no reason is indistinguishable from an oversight."""
    assert UNCOVERED_SEMANTICS
    for name, reason in UNCOVERED_SEMANTICS.items():
        assert len(reason) > 40, name


@pytest.mark.parametrize("block_type", ("full_attention", "linear_attention"))
def test_both_published_token_mixers_are_covered(block_type):
    """Three layers in four are linear attention and one in four is full
    attention; a fixture that covered only one of them would cover a model the
    published stack does not contain."""
    assert block_type in reference.CONFIG.layer_types
    layer = LAYER_TYPE[block_type].cloned()
    assert {child.name for child in layer.modules} == {"mixer", "moe"}
