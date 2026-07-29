"""Greedy continuation decode against the published Qwen3-1.7B checkpoint.

16 greedy steps, HIR against Hugging Face, compared as token IDs. Both sides step
one token at a time over their own caches, and the weights reach HIR through the
alias table and the converters -- so this is where a published key or layout that
was read wrongly shows up.

Two boundaries. Prefill is Hugging Face's: `ctx_len` is bounded below by 1, so the
kernels express a continuation step and not the prompt before it, and what is
claimed here is continuation decode over a real checkpoint rather than a complete
causal LM. And the checkpoint is a mount named by an environment variable with no
default; nothing here downloads anything, and absent the variable this skips.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.models.qwen3_1_7b import config
from tests.models.qwen3_1_7b.hf_alias import hf_alias
from tests.models.qwen3_1_7b.model import Qwen3_1_7B_Decoder
from tilefoundry.runtime import SafetensorsResource

CKPT_ENV = "TILEFOUNDRY_QWEN3_1_7B_CKPT"

PROMPT = "Explain why the sky appears blue in one sentence."
STEPS = 16
DEV = "cuda"

pytestmark = [
    pytest.mark.l3,
    pytest.mark.skipif(not os.environ.get(CKPT_ENV), reason=f"{CKPT_ENV} is unset"),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="two f32 copies of a 1.7B model"
    ),
]


@pytest.fixture(scope="module")
def mount() -> str:
    """The checkpoint directory, refused unless it is the model this package pins.

    Any readable directory would otherwise satisfy the variable, and a mismatched
    one would report agreement about a model nobody asked for.
    """
    ckpt = os.environ[CKPT_ENV]
    published = json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8"))
    shape = config.REAL

    assert published["model_type"] == "qwen3", published["model_type"]
    assert (
        published["hidden_size"], published["num_hidden_layers"],
        published["vocab_size"], published["head_dim"],
        published["intermediate_size"], published["num_key_value_heads"],
    ) == (
        shape.hidden, shape.n_layers, shape.vocab, shape.head_dim,
        shape.intermediate, shape.n_kv_heads,
    )
    return ckpt


@pytest.fixture(scope="module")
def hugging_face(mount):
    """The published model and tokenizer, at the f32 the kernels are authored at."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(mount)
    model = AutoModelForCausalLM.from_pretrained(mount, dtype=torch.float32)
    return tokenizer, model.to(DEV).eval()


@pytest.fixture(scope="module")
def loaded(mount, tmp_path_factory):
    """The decoder root with the published weights bound, through `prepare`."""
    root = Qwen3_1_7B_Decoder.cloned()
    raw = SafetensorsResource(
        mount, device="cpu", alias=hf_alias(config.REAL), dtype=torch.float32
    )
    prepared = tmp_path_factory.mktemp("qwen3_1_7b_prepared")
    root.prepare(raw, str(prepared), device="cpu")
    return root.load(SafetensorsResource(str(prepared), device=DEV))


def _hf_greedy(model, prompt_ids):
    """Hugging Face's own 16 greedy continuations, and the prompt's cache.

    Stepped by hand rather than through `generate`, so both sides run 16 steps
    whatever the tokens are and no `generation_config` field decides what greedy
    means here.
    """
    with torch.no_grad():
        prefill = model(prompt_ids[:, :-1], use_cache=True)
        context = tuple(
            (layer.keys.transpose(1, 2).contiguous(), layer.values.transpose(1, 2).contiguous())
            for layer in prefill.past_key_values.layers
        )

        cache, token, tokens = prefill.past_key_values, prompt_ids[:, -1:], []
        for _ in range(STEPS):
            step = model(token, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            token = torch.argmax(step.logits[:, -1], dim=-1, keepdim=True)
            tokens.append(int(token))
    return context, tokens


def test_greedy_continuation_matches_hugging_face_token_for_token(
    hugging_face, loaded
) -> None:
    """16 IDs, both sides, from the same prompt cache and the prompt's last token."""
    tokenizer, model = hugging_face
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEV)
    assert prompt_ids.shape[1] >= 2, "a continuation needs a token of context"

    context, want = _hf_greedy(model, prompt_ids)
    cos, sin = config.rope_caches(model.config, config.REAL.max_pos, device=DEV)
    scale = torch.full((1, 1, 1, 1), model.model.layers[0].self_attn.scaling, device=DEV)

    caches, token = context, prompt_ids[0, -1:]
    got = []
    for offset in range(STEPS):
        pos_ids = torch.tensor(
            [prompt_ids.shape[1] - 1 + offset], device=DEV, dtype=torch.int32
        )
        logits, entries = loaded.forward(token, cos, sin, pos_ids, scale, caches)
        caches = loaded.append_cache(caches, entries)
        token = torch.argmax(logits[0]).reshape(1).to(torch.int64)
        got.append(int(token))

    assert got == want, (
        f"diverged at step {next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)}: "
        f"{tokenizer.decode(got)!r} against {tokenizer.decode(want)!r}"
    )
