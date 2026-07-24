"""HF reference path for Qwen3.5-35B-A3B: correctness anchor + HF-eager decode-TPS
baseline (P0a plan Q0, docs/plans/agent-kernel-loop/P1-qwen35-runtime-400tps.md).

Text-only. Real checkpoint: /data2/models/Qwen3.5-35B-A3B/ (~70GB bf16,
ModelScope download). GPU: this module never touches `CUDA_VISIBLE_DEVICES`
itself -- the caller pins it (task: GPU1); `device="cuda"` here just means
"whatever GPU index 0 is after that env var's remapping".

Loading gotcha (see `load_text_model` docstring): the on-disk checkpoint is the
full VLM (`Qwen3_5MoeForConditionalGeneration`) and stores every decoder tensor
under a `model.language_model.*` prefix. The standalone text-only class
`Qwen3_5MoeForCausalLM` (what `AutoModelForCausalLM.from_pretrained` resolves
to for this `model_type`) wires its decoder as `self.model` with NO
`language_model` infix. Loading straight into that class -- or via
`AutoModelForCausalLM` -- silently leaves the entire decoder randomly
initialized (reported as merely a warning, not an error): every
`model.layers.*`/`model.embed_tokens.*`/`model.norm.*` key comes up "missing"
while the real `model.language_model.*` tensors sit in the checkpoint unused.
Verified empirically with a meta-device state_dict key diff (both classes
built from the real config.json, no data needed) -- see the task report.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextModel,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class TextRef:
    """Text decoder + lm_head extracted from the full VLM checkpoint.

    `text_model` is `full.model.language_model` (a `Qwen3_5MoeTextModel`) and
    `lm_head` is `full.lm_head` (top-level in the checkpoint, not nested under
    `language_model`) -- see `load_text_model`. Calling `text_model` directly
    (bypassing the VLM's own `forward`) is exactly what
    `Qwen3_5MoeForCausalLM.forward` does internally, so this reproduces
    text-only decoding exactly, with no VLM-specific (image/video rope-index)
    code path ever exercised.
    """

    text_model: Qwen3_5MoeTextModel
    lm_head: torch.nn.Linear
    config: Qwen3_5MoeTextConfig
    eos_token_ids: list[int]
    device: torch.device


def load_text_model(ckpt_dir: str, device: str = "cuda") -> tuple[TextRef, PreTrainedTokenizerBase]:
    """Loads the Qwen3.5-35B-A3B text decoder + tokenizer. bf16,
    `attn_implementation="eager"` (matches the "HF eager" baseline this module
    reports -- no SDPA/flash-attention. The linear-attention layers also have
    no fused-kernel path available in this env: `causal_conv1d` and
    `flash_linear_attention` are both uninstalled here, verified via
    `is_causal_conv1d_available()` / `is_flash_linear_attention_available()`
    both returning `False`, so every layer -- full-attention AND
    linear-attention -- runs its plain-torch fallback; "eager" applies to the
    whole model, not just the 10 full-attention layers).

    Returns `(TextRef, tokenizer)`. Frees the vision tower
    (`full.model.visual`, ~0.8GB bf16) right after loading since the text path
    never touches it -- the one place this honors "save VRAM where free";
    loading the full VLM class first is what makes the weight loading itself
    correct (see module docstring), so that part is not skipped.
    """
    full = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        ckpt_dir, dtype=torch.bfloat16, attn_implementation="eager"
    )
    full = full.to(device).eval()

    del full.model.visual
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    text_model = full.model.language_model
    lm_head = full.lm_head
    text_config = full.config.get_text_config()

    try:
        gen_cfg = GenerationConfig.from_pretrained(ckpt_dir)
        eos_ids = gen_cfg.eos_token_id
    except OSError:
        eos_ids = text_config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    ref = TextRef(
        text_model=text_model,
        lm_head=lm_head,
        config=text_config,
        eos_token_ids=list(eos_ids),
        device=torch.device(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    return ref, tokenizer


@torch.no_grad()
def _prefill(ref: TextRef, input_ids: torch.Tensor) -> tuple[torch.Tensor, object]:
    """One forward call over the whole prompt; `past_key_values=None` +
    `use_cache=True` makes `Qwen3_5MoeTextModel.forward` create a fresh hybrid
    cache itself (per-layer conv/recurrent state for linear-attention layers,
    standard KV for full-attention layers, dispatched off `config.layer_types`
    -- see `transformers.cache_utils.DynamicCache`).

    Returns `(logits_last, past_key_values)`; `logits_last` is `[1, vocab]`
    (lm_head applied to the last position's hidden state only -- equivalent to
    `Qwen3_5MoeForCausalLM.forward`'s `logits_to_keep=1`, avoids projecting
    every prompt position through the 248320-wide vocab head).
    """
    out = ref.text_model(input_ids=input_ids, use_cache=True, past_key_values=None)
    logits_last = ref.lm_head(out.last_hidden_state[:, -1, :])
    return logits_last, out.past_key_values


@torch.no_grad()
def _decode_step(ref: TextRef, input_ids_1tok: torch.Tensor, past_key_values: object) -> tuple[torch.Tensor, object]:
    """One single-token decode step against an existing cache."""
    out = ref.text_model(input_ids=input_ids_1tok, use_cache=True, past_key_values=past_key_values)
    logits_last = ref.lm_head(out.last_hidden_state[:, -1, :])
    return logits_last, out.past_key_values


def greedy_generate(ref: TextRef, tokenizer: PreTrainedTokenizerBase, prompt: str, max_new: int) -> str:
    """Pure-text greedy decode. Returns only the newly generated continuation
    text (prompt not included); stops early on any of `ref.eos_token_ids`.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(ref.device)
    logits, past = _prefill(ref, input_ids)

    generated: list[int] = []
    for _ in range(max_new):
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id in ref.eos_token_ids:
            break
        generated.append(next_id)
        next_input = torch.tensor([[next_id]], device=ref.device, dtype=input_ids.dtype)
        logits, past = _decode_step(ref, next_input, past)

    return tokenizer.decode(generated, skip_special_tokens=True)


@dataclass
class DecodeTpsResult:
    prefill_s: float
    decode_s: float
    n_tokens: int
    tps: float


def decode_tps(ref: TextRef, tokenizer: PreTrainedTokenizerBase, prompt: str, n: int = 64) -> DecodeTpsResult:
    """HF-eager decode-TPS baseline: prefill once (timed separately, not
    counted in `tps`), then `n` greedy decode steps timed with CUDA events
    (device-side; excludes host-side script overhead before/after, but each
    step still pays its real per-launch dispatch cost AND the `.item()`
    host-device sync greedy decoding inherently needs to read back the sampled
    token id -- both are exactly the overhead CUDA-graph capture in the
    runtime is meant to remove, so leaving them in is the point of an "eager"
    baseline).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(ref.device)
    is_cuda = ref.device.type == "cuda"

    if is_cuda:
        torch.cuda.synchronize(ref.device)
        pre_start, pre_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        pre_start.record()
    logits, past = _prefill(ref, input_ids)
    if is_cuda:
        pre_end.record()
        torch.cuda.synchronize(ref.device)
        prefill_s = pre_start.elapsed_time(pre_end) / 1000.0
    else:
        prefill_s = float("nan")

    next_id = int(torch.argmax(logits, dim=-1).item())
    next_input = torch.tensor([[next_id]], device=ref.device, dtype=input_ids.dtype)

    if is_cuda:
        start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(ref.device)
        start_evt.record()
    for _ in range(n):
        logits, past = _decode_step(ref, next_input, past)
        next_id = int(torch.argmax(logits, dim=-1).item())
        next_input = torch.tensor([[next_id]], device=ref.device, dtype=input_ids.dtype)
    if is_cuda:
        end_evt.record()
        torch.cuda.synchronize(ref.device)
        decode_s = start_evt.elapsed_time(end_evt) / 1000.0
    else:
        decode_s = float("nan")

    return DecodeTpsResult(prefill_s=prefill_s, decode_s=decode_s, n_tokens=n, tps=n / decode_s)


__all__ = [
    "TextRef",
    "DecodeTpsResult",
    "load_text_model",
    "greedy_generate",
    "decode_tps",
]
