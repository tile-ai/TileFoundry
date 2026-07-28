"""One Qwen3.5-35B-A3B decoder layer as a tilefoundry IR Module tree, over free
``config``, ``mixer_module`` and ``moe_module`` names -- not importable on its
own, load it with ``tests.models.loader.load_model`` (see ``../decoder_layer.py``).

The layer is two residual additions around two blocks, and which blocks they are
is what ``layer_types`` decides. So the token mixer is *injected*: this one file
is both published layer shapes, and the thing that distinguishes them is the
child handed in rather than a branch written here. That matters beyond tidiness
-- a branch would make the two layers one Module with two behaviours, and the
whole point of the published stack is that they are different kernels.

The mixer and the MoE block are child Modules rather than functions of this one
because an HIR Function may only call a Function its own Module owns. Nesting
keeps each in its own execution domain, which is also what lets analysis and
scheduling select the mixer alone.

The walk is a plain orchestration method rather than a ``@func`` body, following
``tests/models/deepseek_v4_flash/model/decoder_layer.py``: what it composes are
Modules, and a Function cannot call across domains.

Both mixers hand their state back through this boundary unchanged. What that
state *is* differs -- a key and a value for the full-attention layer, a
convolution column and a recurrent matrix for the linear-attention one -- and
this layer deliberately does not know which: it is the caller that owns the
state, so the caller is where knowing what to do with it belongs.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings

S = 1


@module(entry="residual_add")
class Qwen3_5DecoderLayer:
    mixer = mixer_module  # noqa: F821 -- injected by the loader
    moe = moe_module  # noqa: F821 -- injected by the loader

    @func
    def residual_add(
        a: Tensor[(1, S, config.hidden), config.dt],
        b: Tensor[(1, S, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        return tf.add(a, b)

    def forward(self, hidden, mixer_args, moe_args):
        """One decode step: mixer + residual, then MoE + residual.

        Mirrors ``Qwen3_5MoeDecoderLayer.forward``. The two pre-norms are not
        here because each block fuses its own -- the mixer fuses
        ``input_layernorm`` and the MoE block fuses
        ``post_attention_layernorm``, so each fused kernel lines up with one
        Hugging Face pre-norm-then-block composition.

        What comes back is the layer output and whatever state the mixer
        produced, passed through untouched for the caller to advance.
        """
        mixed, *state = self.mixer(hidden, *mixer_args)
        attended = self.residual_add(hidden, mixed)
        expert_out = self.moe(attended, *moe_args)
        return self.residual_add(attended, expert_out), tuple(state)
