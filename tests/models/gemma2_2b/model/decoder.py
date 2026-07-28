"""The complete Gemma-2-2B decoder as one tilefoundry IR Module tree, over free
``config`` and ``decoder_layers`` names -- not importable on its own, load it with
``tests.models.loader.load_model`` (see ``../decoder.py``).

One layer's verified behaviour does not add up to the stack behaving. Layer
order, the final norm, and the residual thread running between layers are only
observable when every layer is present, and a per-layer comparison against
Hugging Face passes whether or not they are right. So the decoder is its own
Module: the layers are its children, in order, and its boundary is hidden states
in and hidden states out.

The layer walk is a plain orchestration method rather than a ``@func`` body,
following ``tests/models/qwen2_5_1_5b/model/decoder.py``. A ``@func`` would have
to carry 26 layers' key and value increments out of a loop, and nothing writes
into a slot of a carried tensor -- so the walk would have to be unrolled into one
body, which states the same 26 layers with none of the structure that says they
are 26 of the same thing. As children they stay separately addressable, which is
also what lets analysis and scheduling select one layer rather than the whole
stack.

``final_rms_norm`` is ``Gemma2Model.norm``, and like every other norm in this
model it is ``normed * (1.0 + weight)`` on Hugging Face's side, so the weight
handed to :meth:`bind_final_norm` is expected pre-adjusted by
``config.rms_gamma``.

Token embedding and the output projection are deliberately absent. They sit
either side of this boundary, they are the only vocabulary-sized weights in the
model, and nothing here needs them to say whether the stack is right. Gemma-2's
embedding also scales by ``sqrt(hidden_size)`` and its logits carry their own
``final_logit_softcapping``; both belong to the pieces that are not here.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies

S = 1


@module(entry="final_rms_norm")
class Gemma2_2B_Decoder:
    """The ordered layer stack plus the norm that closes it."""

    layers = decoder_layers  # noqa: F821 — injected by the loader

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_final: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `Gemma2Model.norm`, applied once after the last layer.
        # `gamma_final` is pre-adjusted to `1.0 + weight` test-side.
        return tf.rms_norm(hidden, gamma_final)

    def forward(self, hidden, cos_cache, sin_cache, pos_ids, scale, weights, caches):
        """One decode step through every layer, then the final norm.

        *weights* and *caches* are per layer, in layer order. What comes back is
        the normalised hidden state and each layer's own cache entry, for the
        caller to append -- the same division the single layer makes, kept at the
        stack's boundary so the caller owns the cache at exactly one place.
        """
        if len(weights) != len(self.modules) or len(caches) != len(self.modules):
            raise ValueError(
                f"decoder has {len(self.modules)} layers but was given "
                f"{len(weights)} weight sets and {len(caches)} caches"
            )
        entries = []
        for layer, layer_weights, (k_cache, v_cache) in zip(self.modules, weights, caches):
            (
                gamma_in, w_q, w_k, w_v, w_o, gamma_post_attn,
                gamma_pre_ff, w_gate, w_up, w_down, gamma_post_ff,
            ) = layer_weights
            hidden, k_new, v_new = layer(
                hidden, gamma_in, w_q, w_k, w_v, cos_cache, sin_cache, pos_ids,
                k_cache, v_cache, scale, w_o, gamma_post_attn,
                gamma_pre_ff, w_gate, w_up, w_down, gamma_post_ff,
            )
            entries.append((k_new, v_new))
        return self.final_rms_norm(hidden, self._gamma_final), tuple(entries)

    def bind_final_norm(self, gamma_final):
        """Hold the final norm's weight, which `forward` does not take per layer."""
        object.__setattr__(self, "_gamma_final", gamma_final)
        return self
