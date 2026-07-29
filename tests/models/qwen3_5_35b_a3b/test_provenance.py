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

from tests.models.qwen3_5_35b_a3b import config
from tests.models.qwen3_5_35b_a3b.model import (
    LAYER_TYPE,
    Qwen3_5Decoder,
    Qwen3_5FullAttention,
    Qwen3_5FullAttnLayer,
    Qwen3_5LinearAttention,
    Qwen3_5LinearAttnLayer,
    Qwen3_5MoE,
)
from tilefoundry.ir.core.module import select

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── where the numbers came from ─────────────────────────────────────────


def test_the_shape_is_the_published_one_field_by_field():
    """Every field of ``REAL`` traces to a field of the published configuration.

    ``max_ctx`` and ``dt`` are the two that do not, and they are named here
    rather than exempted silently: one is the authoring envelope the kernels were
    written to and the other the dtype they are compared at. Neither is a claim
    about the checkpoint.
    """
    published = config.published_text_config()
    shape = config.REAL

    assert (shape.hidden, shape.vocab) == (
        published["hidden_size"], published["vocab_size"]
    )
    assert (shape.n_q_heads, shape.n_kv_heads, shape.head_dim) == (
        published["num_attention_heads"],
        published["num_key_value_heads"],
        published["head_dim"],
    )
    assert (shape.n_experts, shape.top_k) == (
        published["num_experts"], published["num_experts_per_tok"]
    )
    assert (shape.moe_intermediate, shape.shared_intermediate) == (
        published["moe_intermediate_size"],
        published["shared_expert_intermediate_size"],
    )
    assert (
        shape.gdn_n_k_heads, shape.gdn_head_k_dim,
        shape.gdn_n_v_heads, shape.gdn_head_v_dim, shape.gdn_conv_kernel,
    ) == (
        published["linear_num_key_heads"], published["linear_key_head_dim"],
        published["linear_num_value_heads"], published["linear_value_head_dim"],
        published["linear_conv_kernel_dim"],
    )
    assert shape.layer_types == tuple(published["layer_types"])
    assert shape.n_layers == published["num_hidden_layers"] == len(shape.layer_types)
    assert shape.rope_theta == published["rope_parameters"]["rope_theta"]
    assert (
        shape.partial_rotary_factor
        == published["rope_parameters"]["partial_rotary_factor"]
    )
    assert shape.mtp_n_layers == published["mtp_num_hidden_layers"]

    # The two fields that are ours, not the checkpoint's.
    declared = {field for field in shape.__dataclass_fields__}
    ours = {"max_ctx", "dt"}
    assert ours <= declared
    assert shape.max_ctx < shape.max_pos, (
        "the authoring envelope must be inside the published position limit"
    )


def test_the_depended_on_fields_still_digest_to_what_was_recorded():
    """A shape change has to be a visible edit to the recorded digest.

    Retyping a dimension is the failure this guards: it leaves the package
    self-consistent and no longer describing the checkpoint it names.
    """
    assert config.fields_digest() == config.FIELDS_DIGEST


def test_the_provenance_names_a_revision_and_not_only_a_branch():
    """A branch name would let this file's provenance move under it."""
    assert config.SOURCE_URL.startswith("https://huggingface.co/Qwen/Qwen3.5-35B-A3B")
    assert len(config.SOURCE_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in config.SOURCE_REVISION)
    assert len(config.SOURCE_SHA256) == 64
    assert config.TEXT_CONFIG_PATH.is_file()


def test_hugging_face_accepts_the_published_fields_verbatim():
    """The published dictionary is handed over as it is, and is not rejected.

    ``Qwen3_5MoeTextConfig`` is a strict dataclass, so this is the check that the
    installed transformers understands *this* configuration rather than a subset
    of it that happens to overlap.
    """
    built = config.build_hf_config()
    published = config.published_text_config()

    assert built.model_type == published["model_type"] == "qwen3_5_moe_text"
    for field in ("hidden_size", "head_dim", "num_experts", "num_experts_per_tok",
                  "linear_num_value_heads", "num_hidden_layers"):
        assert getattr(built, field) == published[field], field


def test_a_truncated_stack_still_names_the_layer_type_it_was_asked_for():
    """Layers are built at the lowest published index of their type.

    A decoder layer reads its own type out of ``layer_types[layer_idx]``, so the
    index is how the type is chosen. Building a one-layer config and asking for a
    full-attention layer would silently produce a linear-attention one.
    """
    shape = config.REAL
    for block_type in ("linear_attention", "full_attention"):
        index = shape.first_layer_of(block_type)
        built = config.build_hf_config(layers=index + 1)
        assert built.layer_types[index] == block_type
    assert shape.first_layer_of("full_attention") == shape.full_attention_interval - 1


def test_the_stack_is_the_published_order_and_its_layers_are_independent():
    """The stack is `layer_types` in order, and each layer is an independent copy.

    A stack built from a restated pattern, or from layers sharing Functions, would
    pass every component test in this package and fail here.
    """
    shape = config.REAL
    stack = Qwen3_5Decoder
    assert len(stack.modules) == shape.n_layers
    assert [child.name for child in stack.modules[:3]] == ["layer0", "layer1", "layer2"]

    for index, kind in enumerate(shape.layer_types):
        mixer = select(stack, f"layer{index}.mixer")
        entry = LAYER_TYPE[kind].modules[0].entry
        assert mixer.entry == entry, f"layer{index} is not a {kind} layer"

    first, second = stack.modules[0], stack.modules[1]
    assert first.lookup("residual_add") is not second.lookup("residual_add")
    assert first.modules[0].lookup("conv_step") is not second.modules[0].lookup("conv_step")

    selected = select(stack, "layer0.moe.moe")
    assert selected.name == "moe"
    assert selected.entry_function().name == "moe"

    assert "forward" in stack.methods
    with pytest.raises(ValueError, match=f"{shape.n_layers} layers but was given 0"):
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

    shape = config.REAL
    total = 64
    cos, sin = config.rope_caches(total=total, device=DEV)
    assert cos.shape == (total, shape.rotary_dim), (
        "the caches are rotary_dim wide, not head_dim: partial_rotary_factor "
        f"{shape.partial_rotary_factor} of {shape.head_dim}"
    )

    with torch.device(DEV):
        rotary = Qwen3_5MoeTextRotaryEmbedding(config.build_hf_config(layers=1))
    frequencies = torch.outer(
        torch.arange(total, device=DEV).float(), rotary.inv_freq
    )
    plain = torch.cat([frequencies, frequencies], dim=-1)

    assert (plain.cos() - cos).abs().max().item() < 1e-9
    assert (plain.sin() - sin).abs().max().item() < 1e-9


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

    assert config.REAL.mtp_n_layers == 1

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
    "Qwen3_5MoE": ("routing", "routed_experts", "shared_expert", "moe"),
    "Qwen3_5FullAttention": ("partial_rope", "full_attention"),
    "Qwen3_5LinearAttention": ("linear_attention",),
    "Qwen3_5FullAttnLayer": ("residual_add",),
    "Qwen3_5LinearAttnLayer": ("residual_add",),
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
    "Qwen3_5Decoder": ("embed", "final_rms_norm", "lm_head"),
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
        "`Qwen3_5Decoder` composes the published order and orchestrates it, so the "
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
        Qwen3_5MoE,
        Qwen3_5FullAttention,
        Qwen3_5LinearAttention,
        Qwen3_5FullAttnLayer,
        Qwen3_5LinearAttnLayer,
        Qwen3_5Decoder,
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
    assert block_type in config.REAL.layer_types
    layer = LAYER_TYPE[block_type].cloned()
    assert {child.name for child in layer.modules} == {"mixer", "moe"}
