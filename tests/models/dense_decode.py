"""How a dense decoder's decode step is drawn, stated once instead of per package.

The four dense packages drew their oracle the same way: seed the weights, seed the
activations, draw one more hidden state than the context, split it into the context
and the token being decoded, build the KV cache from the context, and place the
decoded token immediately after it. Written out per package, that came to four
copies agreeing to within a handful of lines -- and the lines they differed on were
the ones that matter, so the agreement was doing no work while the differences were
easy to lose.

What stays in each package is what makes its oracle *its* oracle: which Hugging
Face layer, which weights its kernel takes and in what order, and the boundary its
`ReferenceCase` declares. This module only knows how to draw a step and where to
ask for the answer.

Weights do not appear here at all. Each package loads its own Module from its own
resource and hands the ``LoadedModule`` in; this module knows only how to draw the
activations and where to ask for the answer, so no canonical name and no weight
order lives in one shared file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def _self_attn_scaling(layer) -> float:
    return layer.self_attn.scaling


def _layers(model):
    """A Hugging Face causal LM's decoder layers."""
    return model.model.layers


@dataclass(frozen=True)
class DenseDecode:
    """One dense decoder package's declaration of how to draw its oracle.

    Every field is something this module calls. It used to take the package's
    ``config`` module and reach into it for whichever name it wanted, which made
    that module an interface nobody had written down -- a package satisfied it by
    happening to export ten right-shaped names. Naming the callables is the same
    dependency stated where it can be read, and it is what let the hand-copied
    dimensions behind ``config.REAL`` be replaced by each model's published
    ``config.json`` without this file knowing.

    ``load_layer`` and ``load_decoder`` are the package's own: given its Hugging
    Face layer or stack, each returns the ``LoadedModule`` to run. Which canonical
    name reads which Hugging Face tensor is the package's statement, not this
    module's.
    """

    #: The published hidden size, the dtype the checkpoint publishes, and the
    #: caches at the published context limit. The dtype is here because every
    #: draw this module makes has to be at it: an oracle built in f32 against
    #: kernels declared at the checkpoint's bf16 would be comparing two
    #: precisions, and calling the difference a tolerance.
    hidden_size: int
    dtype: object
    rope_caches: Callable[[str], tuple[torch.Tensor, torch.Tensor]]
    #: Hugging Face's own layer and stack, drawn at a seed.
    build_layer: Callable[..., object]
    build_decoder: Callable[..., object]
    #: What Hugging Face would hold, and what it would answer.
    context_kv: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    decode_reference: Callable[..., torch.Tensor]
    decoder_context_kv: Callable[..., list]
    decoder_decode_reference: Callable[..., torch.Tensor]
    load_layer: Callable[[object], object]
    load_decoder: Callable[[object], object]
    #: Seeds, named so a change to either is a visible change to the reference.
    weight_seed: int = 0
    activation_seed: int = 1
    #: Small enough to keep the oracle's full-sequence forward cheap, and not a
    #: power of the head count, so an index arithmetic error cannot coincide with
    #: a head boundary.
    ctx_len: int = 24
    #: Whatever runs this states its own device; the oracle runs on the CPU.
    device: str = "cpu"
    scale_of: Callable[[object], float] = _self_attn_scaling
    #: Runtime values this model's signature takes after the weights. MiniCPM3
    #: scales each residual by a depth-dependent constant, which is a value its
    #: step is given rather than a weight it holds.
    trailing: Callable[[object, str], tuple] = lambda layer, device: ()
    #: A package may extend either drawn step with a property derived from what it
    #: already carries -- Gemma2's pure-attention argument projection, MiniCPM3's
    #: named residual scale. Deriving beats assembling a second time: one
    #: parameter order stays one statement.
    layer_step_class: type = None
    stack_step_class: type = None


@dataclass(frozen=True)
class LayerStep:
    """One drawn step of a single layer: the evaluator's arguments and the layer
    behind them.

    Both travel together on purpose. The oracle is a Hugging Face layer with random
    weights, so handing back only tensors would leave a caller free to score them
    against a differently initialised layer.
    """

    args: tuple[torch.Tensor, ...]
    loaded: object
    layer: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor


@dataclass(frozen=True)
class StackStep:
    """One drawn step of the complete decoder, and the stack behind it."""

    model: object
    loaded: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    caches: list
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    pos_ids: torch.Tensor
    scale: torch.Tensor
    trailing: tuple = ()

    @property
    def args(self) -> tuple:
        """What the runner passes on, for a wiring check to count."""
        return (
            self.hidden_new, self.cos_cache, self.sin_cache, self.pos_ids,
            self.scale, *self.trailing, self.caches,
        )


def _split_hidden(spec: DenseDecode, ctx_len: int, device: str):
    """The context and the token that follows it, drawn together and seeded."""
    torch.manual_seed(spec.activation_seed)
    drawn = (torch.randn(1, ctx_len + 1, spec.hidden_size, device=device) * 0.1).to(
        spec.dtype
    )
    return drawn[:, :ctx_len], drawn[:, ctx_len:]


def _positions(ctx_len: int, device: str) -> torch.Tensor:
    # The token being decoded sits immediately after the context.
    return torch.tensor([ctx_len], device=device, dtype=torch.int32)


def layer_step(
    spec: DenseDecode, *, ctx_len: int | None = None, device: str | None = None
) -> LayerStep:
    """One deterministic decode step of one layer over a *ctx_len*-token context."""
    ctx_len = spec.ctx_len if ctx_len is None else ctx_len
    device = spec.device if device is None else device

    layer = spec.build_layer(seed=spec.weight_seed, device=device)
    cos_cache, sin_cache = spec.rope_caches(device)
    scale = torch.full(
        (1, 1, 1, 1), spec.scale_of(layer), device=device, dtype=spec.dtype
    )
    hidden_ctx, hidden_new = _split_hidden(spec, ctx_len, device)
    k_cache, v_cache = spec.context_kv(layer, hidden_ctx, device=device)

    return (spec.layer_step_class or LayerStep)(
        args=(
            hidden_new,
            cos_cache,
            sin_cache,
            _positions(ctx_len, device),
            k_cache,
            v_cache,
            scale,
            *spec.trailing(layer, device),
        ),
        loaded=spec.load_layer(layer),
        layer=layer,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        k_cache=k_cache,
        v_cache=v_cache,
    )


def layer_oracle(spec: DenseDecode, drawn: LayerStep) -> torch.Tensor:
    """What Hugging Face's own layer produces for the same drawn step."""
    return spec.decode_reference(drawn.layer, drawn.hidden_ctx, drawn.hidden_new)


def appended_cache(spec: DenseDecode, drawn: LayerStep):
    """The cache the step's caller should hold afterwards.

    Built the way the input cache was, over the context with the decoded token
    appended: the kernel's returned key and value are correct exactly when
    appending them reproduces this.
    """
    return spec.context_kv(
        drawn.layer,
        torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1),
        device=drawn.hidden_ctx.device.type,
    )


def stack_step(
    spec: DenseDecode, *, ctx_len: int | None = None, device: str = "cuda"
) -> StackStep:
    """One decode step of the complete decoder over a *ctx_len*-token context.

    The whole stack, so the drawn problem is per layer: each layer's weights and
    each layer's own cache, in layer order. Built at the production layer count,
    because layer order and the residual thread between layers are what this
    boundary exists to observe.
    """
    ctx_len = spec.ctx_len if ctx_len is None else ctx_len
    model = spec.build_decoder(seed=spec.weight_seed, device=device)
    hidden_ctx, hidden_new = _split_hidden(spec, ctx_len, device)
    cos_cache, sin_cache = spec.rope_caches(device)
    first_layer = _layers(model)[0]
    return (spec.stack_step_class or StackStep)(
        model=model,
        loaded=spec.load_decoder(model),
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        caches=spec.decoder_context_kv(model, hidden_ctx, device=device),
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        pos_ids=_positions(ctx_len, device),
        scale=torch.full(
            (1, 1, 1, 1), spec.scale_of(first_layer), device=device, dtype=spec.dtype
        ),
        trailing=spec.trailing(first_layer, device),
    )


def run_stack(spec: DenseDecode, drawn: StackStep):
    """The complete decoder over *drawn*, through the Evaluator.

    ``decode_hidden`` rather than ``forward``: this boundary starts from a hidden
    state, and the root's ``forward`` is the whole decode step from token ids. The
    weights are already bound, so what is passed is activations.
    """
    return drawn.loaded.decode_hidden(*drawn.args)


def stack_oracle(spec: DenseDecode, drawn: StackStep) -> torch.Tensor:
    """What Hugging Face's own stack produces for the same drawn step."""
    return spec.decoder_decode_reference(
        drawn.model, drawn.hidden_ctx, drawn.hidden_new
    )


__all__ = [
    "DenseDecode",
    "LayerStep",
    "StackStep",
    "appended_cache",
    "layer_oracle",
    "layer_step",
    "run_stack",
    "stack_oracle",
    "stack_step",
]
