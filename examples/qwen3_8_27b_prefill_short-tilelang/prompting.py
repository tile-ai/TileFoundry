"""The one rule that turns a prompt into token ids, shared by run.py, the
transformers reference and both benchmark workers.

`--prompt-tokens N` repeats the encoded prompt cyclically (or cuts it) to exactly
N ids, so every length is served from the same text. No chat template is applied:
the ids are the tokenizer's plain encoding of the text.
"""
from __future__ import annotations

import math

import torch

LENGTHS = (128, 256, 512, 1024)


def prompt_ids(tokenizer, prompt: str, n_tokens: int | None = None) -> list[int]:
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("the prompt encodes to no tokens")
    if n_tokens is None:
        return ids
    reps = -(-n_tokens // len(ids))
    return (ids * reps)[:n_tokens]


def rope_tables(n_tokens: int, config: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin rows for positions 0..n-1, as `Qwen3_5TextRotaryEmbedding` builds them
    for text: f32 inv_freq, f32 phases, cat(freqs, freqs), cos/sin, landed in bf16.
    For text the three mrope sections read the same position, so the interleave
    is the identity."""
    rope = config["rope_parameters"]
    dim = int(config["head_dim"] * rope["partial_rotary_factor"])
    inv = 1.0 / (rope["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    pos = torch.arange(n_tokens, device=device, dtype=torch.float)
    freqs = (inv[None, :, None] @ pos[None, None, :]).transpose(1, 2)[0]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


__all__ = ["LENGTHS", "prompt_ids", "rope_tables", "math"]
