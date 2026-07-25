"""Model-agnostic decode loop shared by the model fixtures: the caller owns the
caches, the model owns ``init_caches`` / ``prepare_inputs_for_generation``.
"""
from __future__ import annotations

__all__ = ["generate"]


def _module_name(m) -> str:
    return getattr(m, "name", None) or type(m).__name__


def _require_hook(m, hook: str) -> None:
    if not hasattr(m, hook):
        raise TypeError(f"generate: module {_module_name(m)!r} has no {hook!r} hook")


def generate(m, input_ids, max_new_tokens: int, *, device: str = "cuda"):
    """Run *max_new_tokens* decode steps of *m*, threading ``past_key_values``
    from step to step. Returns ``(logits_per_step, past_key_values)``; sampling
    and stopping are the caller's."""
    _require_hook(m, "init_caches")
    _require_hook(m, "prepare_inputs_for_generation")

    past_key_values = m.init_caches(device=device)
    logits_per_step = []
    for t in range(max_new_tokens):
        args = m.prepare_inputs_for_generation(input_ids, t, past_key_values, device=device)
        logits, past_key_values = m(*args)
        logits_per_step.append(logits)
    return tuple(logits_per_step), past_key_values
