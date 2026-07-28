"""Building a decode oracle out of a Hugging Face model, without its cache.

Every model in the corpus states the same decode contract: one token per step,
the active context length as the only dimension left as a range, and the KV cache
passed as explicit tensors in and out. The reference for that contract is
constructed the same way every time, and the construction is the part worth
stating once -- it is a claim about Hugging Face, not about any one model.

Two facts hold it up, both measured rather than assumed:

- A layer run over the whole sequence under a causal mask, with the last position
  kept, equals a cached one-token step. Causality makes that position's output
  depend on exactly the context before it, so the equality is structural and not
  a coincidence of dimensions.
- Keys and values built by running the layer's own norm, projections and rotary
  embedding over the context are what its cache would have held. Not
  approximately: the same tensors, through the same modules, in the same order.

Together they mean no ``Cache`` object is constructed on either side. That matters
because the kernels under test own their cache explicitly; a reference that
borrowed Hugging Face's caching would be checking one cache implementation
against another rather than checking the computation.

What is *not* here is how one attention module turns normed hidden states into a
key and a value. That genuinely differs -- Qwen3 normalises per head and Qwen2
does not, Qwen2 carries projection biases and Qwen3 does not -- so each model
states its own, and passes it in.
"""

from __future__ import annotations

from typing import Callable

import torch


def causal_mask(total: int, device: str = "cpu", dtype=None) -> torch.Tensor:
    """Additive mask ``[1, 1, total, total]``: 0 where a query may attend a key.

    Every position attends every earlier one and itself. A decode step's own
    query sits last, so under this mask it sees the whole context -- which is
    why the step's kernels need no mask of their own.
    """
    positions = torch.arange(total, device=device)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total)
    return mask if dtype is None else mask.to(dtype)


def rope_caches(rotary_class, cfg, total: int, device: str = "cpu", dtype=None):
    """Full cos / sin caches ``[total, head_dim]`` from a model's rotary embedding.

    Row ``p`` is the embedding for absolute position ``p``, so gathering by
    position ids reproduces what the model's own attention applies.
    """
    rotary = rotary_class(cfg).to(device)
    reference = torch.zeros(1, total, cfg.hidden_size, device=device)
    cos, sin = rotary(reference, torch.arange(total, device=device).unsqueeze(0))
    cos, sin = cos[0], sin[0]
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def run_layers(layers, hidden, cos, sin, mask, final_norm=None):
    """*hidden* through every layer in order, cache-free, optionally normed.

    The layers are walked rather than the model called, because calling the model
    would have it manage a cache. Walking them applies exactly the computation
    under test and nothing around it.
    """
    embeddings = (cos.unsqueeze(0), sin.unsqueeze(0))
    for layer in layers:
        hidden = layer(hidden, position_embeddings=embeddings, attention_mask=mask)
    return hidden if final_norm is None else final_norm(hidden)


def decode_reference(layers, hidden_ctx, hidden_new, cos, sin, final_norm=None):
    """What *layers* produce for *hidden_new* decoded after *hidden_ctx*.

    The whole sequence under a causal mask, last position kept.
    """
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    mask = causal_mask(total, hidden_ctx.device.type, hidden_ctx.dtype)
    with torch.no_grad():
        out = run_layers(
            layers, torch.cat([hidden_ctx, hidden_new], dim=1), cos, sin, mask, final_norm
        )
    return out[:, hidden_ctx.shape[1] :, :]


def layer_inputs_over_context(layers, hidden_ctx, cos, sin) -> list[torch.Tensor]:
    """Each layer's own input hidden states for *hidden_ctx*, in layer order.

    Layer ``i``'s cache is built from what layer ``i`` reads, which is what layers
    before it produced -- so the context has to be run through the stack to know
    it. Captured with forward-pre-hooks rather than by asking for a cache, so this
    stays as free of Hugging Face's caching as everything else here.
    """
    captured: list[torch.Tensor | None] = [None] * len(layers)

    def record(index):
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[index] = hidden.detach()
            return None

        return hook

    handles = [
        layer.register_forward_pre_hook(record(index), with_kwargs=True)
        for index, layer in enumerate(layers)
    ]
    try:
        mask = causal_mask(hidden_ctx.shape[1], hidden_ctx.device.type, hidden_ctx.dtype)
        with torch.no_grad():
            run_layers(layers, hidden_ctx, cos, sin, mask)
    finally:
        for handle in handles:
            handle.remove()
    return [hidden for hidden in captured if hidden is not None]


def context_kv(
    layer,
    hidden_ctx: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    key_value_of: Callable,
    apply_rotary: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The cache *layer* would hold for *hidden_ctx*, as explicit tensors.

    *key_value_of* takes the layer and its normed hidden states and returns the
    pre-rotary key and the value, each head-major ``[1, heads, ctx, head_dim]`` --
    it is the one step that differs between models. Rotary is applied here because
    a stored key belongs to the position it was written at, and every model in the
    corpus stores it that way.

    Returned in the kernels' ``[1, ctx_len, n_kv_heads, head_dim]`` layout, not
    Hugging Face's head-major one.
    """
    with torch.no_grad():
        normed = layer.input_layernorm(hidden_ctx)
        key, value = key_value_of(layer, normed)
        # apply_rotary rotates a query/key pair; only the key is wanted.
        _query, key = apply_rotary(key, key, cos.unsqueeze(0), sin.unsqueeze(0))
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


def stack_context_kv(layers, hidden_ctx, cos, sin, *, key_value_of, apply_rotary):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    return [
        context_kv(
            layer, layer_input, cos, sin,
            key_value_of=key_value_of, apply_rotary=apply_rotary,
        )
        for layer, layer_input in zip(
            layers, layer_inputs_over_context(layers, hidden_ctx, cos, sin)
        )
    ]


def randomised(build, seed: int, device: str = "cpu", dtype=None, sigma: float = 0.05):
    """The module *build* returns, on *device*, drawn at *seed*, in eval mode.

    A factory rather than a built module, because where it is built is most of
    the cost. Building on the host and moving initialises every parameter twice
    -- once by the library, once here -- and copies the result across the bus in
    between; for a production-sized stack that is twenty-three seconds against
    two tenths of one. Constructing inside the device's context does none of it.

    Seeded before construction as well as before the draw, so a model whose
    library initialisation consumes randomness still lands on the same weights.
    """
    torch.manual_seed(seed)
    with torch.device(device):
        module = build()
    module = module.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, sigma)
    return module if dtype is None else module.to(dtype)


def linear_weight(linear) -> torch.Tensor:
    """HF ``nn.Linear.weight`` ``[out, in]`` -> the kernels' ``[1, in, out]``.

    The kernel convention is ``x[1, S, in] @ w[1, in, out]``, so the transpose is
    weight preprocessing and belongs on this side of the boundary. A bias, where a
    model has one, needs no such transpose and is used as it is.
    """
    return linear.weight.t().unsqueeze(0).contiguous()


__all__ = [
    "causal_mask",
    "context_kv",
    "decode_reference",
    "layer_inputs_over_context",
    "linear_weight",
    "randomised",
    "rope_caches",
    "run_layers",
    "stack_context_kv",
]
