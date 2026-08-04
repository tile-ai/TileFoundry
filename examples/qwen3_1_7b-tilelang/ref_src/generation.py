"""Autoregressive decode: one token per step; the caller owns the state.

The orchestrator drives a causal-LM Module's orchestration methods. The model
declares a step's activations, while this loop owns sampling and timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass(frozen=True)
class Decoded:
    """The text and decode-only timing from one autoregressive continuation."""

    text: str
    tokens: int
    seconds: float
    prompt_steps: int


def greedy(logits) -> int:
    """Choose the vocabulary entry with the largest logit."""
    return int(torch.argmax(logits).item())


def _sync(device) -> None:
    backend = getattr(torch, torch.device(device).type, None)
    synchronize = getattr(backend, "synchronize", None)
    if synchronize is not None:
        synchronize(device)


def _prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    encoded = tokenizer.encode(prompt)
    return torch.tensor(getattr(encoded, "ids", encoded), dtype=torch.int64)


def decode(loaded, tokenizer, prompt: str, *, max_new: int, sampler=greedy, eos=(), device=None):
    """Decode *prompt* through *loaded* and return its sampled continuation.

    *loaded* may be a ``LoadedModule`` or a runtime twin. Its model supplies the
    four orchestration methods this loop drives, including each step's inputs.
    """
    device = torch.accelerator.current_accelerator() if device is None else device
    prompt_ids = _prompt_ids(tokenizer, prompt).to(device)
    if prompt_ids.numel() == 0:
        raise ValueError("decode needs a prompt that encodes to at least one token")
    prompt_steps = prompt_ids.numel()
    input_ids = torch.empty(prompt_steps + max_new, dtype=torch.int64, device=device)
    input_ids[:prompt_steps] = prompt_ids
    caches = loaded.init_caches(device=device)

    for step in range(prompt_steps):
        args = loaded.prepare_inputs_for_generation(
            input_ids[: step + 1], step, caches, device=device
        )
        logits, fresh = loaded.forward(*args)
        caches = loaded.append_cache(caches, fresh)

    sampler(logits)
    _sync(device)
    started = perf_counter()
    output: list[int] = []
    for step in range(max_new):
        token = int(sampler(logits))
        if token in eos:
            break
        output.append(token)
        input_ids[prompt_steps + step] = token
        args = loaded.prepare_inputs_for_generation(
            input_ids[: prompt_steps + step + 1], prompt_steps + step, caches, device=device
        )
        logits, fresh = loaded.forward(*args)
        caches = loaded.append_cache(caches, fresh)
    _sync(device)
    elapsed = perf_counter() - started
    return Decoded(
        text=tokenizer.decode(output),
        tokens=len(output),
        seconds=elapsed,
        prompt_steps=prompt_steps,
    )


__all__ = ["Decoded", "decode", "greedy"]
