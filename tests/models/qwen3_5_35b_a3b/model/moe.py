"""Qwen3.5-35B-A3B's MoE block as one tilefoundry IR Module, over a free
``config`` name -- not importable on its own, load it with
``tests.models.loader.load_model`` (see ``../moe.py``).

Every layer of the published stack, of either token-mixer type, ends in this
same block, so it is one boundary rather than a detail of two.

Three things happen to a token here, and all three are part of the block:

- it is routed. A linear map to 256 logits, softmaxed in f32, and the best 8
  kept and renormalised to sum to one. The softmax is over all 256 before the
  top-8 is taken, so a change of expert count changes every surviving weight --
  which is why routing is not separable from the expert count the way a dense
  MLP's width is separable from its shape.
- it goes through the 8 experts it selected, and only those. The selection is a
  runtime value, so the expert weights are *gathered* by it: the graph names all
  256 experts and reads 8. That is the whole reason a MoE decode step is a
  different kernel shape from a dense one, and it is what this block is here to
  state.
- it also goes through a shared expert that every token goes through regardless
  of routing, whose contribution is scaled by a scalar the token computes for
  itself (``shared_expert_gate``, a projection to width one, through a sigmoid).
  That parameter appears nowhere in the published configuration -- there is no
  field that implies its existence -- so it has to be read off the architecture,
  and a fixture that reconstructed this block from the configuration alone would
  silently omit it.

The block fuses its preceding RMSNorm, matching the convention of the other
model packages here: each fused kernel then lines up with one Hugging Face
pre-norm-then-block composition.

``silu(x) = x * sigmoid(x)``; there is no standalone silu in the HIR op surface.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings

# One token per step.
S = 1

_H = config.hidden
_E = config.n_experts
_K = config.top_k
_I = config.moe_intermediate
_IS = config.shared_intermediate


@module(entry="moe")
class Qwen3_5MoE:
    @func
    def routing(
        tokens: Tensor[(S, _H), config.dt],
        w_router: Tensor[(_H, _E), config.dt],
    ):
        # HF `Qwen3_5MoeTopKRouter`: softmax over every expert in f32, then the
        # top k, then renormalise. Kept its own function because the selection
        # is the one value in this block that is an index rather than a number,
        # and a router that picked a different 8 would be a different model even
        # if every weight matched.
        logits = tf.cast(tf.matmul(tokens, w_router), dtype="f32")
        probs = tf.softmax(logits, axis=-1)
        top_vals, indices = tf.topk(probs, k=_K, axis=-1)
        denom = tf.reduce(top_vals, axes=(-1,), keepdim=True, kind="sum")
        return tf.cast(tf.div(top_vals, denom), dtype=config.dt), indices

    @func
    def routed_experts(
        tokens: Tensor[(S, _H), config.dt],
        weights: Tensor[(S, _K), config.dt],
        indices: Tensor[(S, _K), "i64"],
        w_gate: Tensor[(_E, _I, _H), config.dt],
        w_up: Tensor[(_E, _I, _H), config.dt],
        w_down: Tensor[(_E, _H, _I), config.dt],
    ) -> Tensor[(S, _H), config.dt]:
        # The gathers are the point: `indices` is a runtime value, so the three
        # expert tensors are indexed by it rather than sliced at a known offset.
        # Each token then runs `top_k` independent SwiGLU experts, batched over
        # the (token, slot) pair, and their outputs are mixed by the routing
        # weights.
        gate_w = tf.gather(w_gate, indices, axis=0)
        up_w = tf.gather(w_up, indices, axis=0)
        down_w = tf.gather(w_down, indices, axis=0)
        token_col = tf.reshape(tokens, new_shape=(S, 1, _H, 1))
        gate = tf.reshape(tf.matmul(gate_w, token_col), new_shape=(S, _K, _I))
        up = tf.reshape(tf.matmul(up_w, token_col), new_shape=(S, _K, _I))
        hidden = tf.mul(tf.mul(gate, tf.sigmoid(gate)), up)
        down = tf.reshape(
            tf.matmul(down_w, tf.reshape(hidden, new_shape=(S, _K, _I, 1))),
            new_shape=(S, _K, _H),
        )
        weighted = tf.mul(down, tf.reshape(weights, new_shape=(S, _K, 1)))
        return tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")

    @func
    def shared_expert(
        tokens: Tensor[(S, _H), config.dt],
        w_shared_gate: Tensor[(_H, _IS), config.dt],
        w_shared_up: Tensor[(_H, _IS), config.dt],
        w_shared_down: Tensor[(_IS, _H), config.dt],
        w_shared_scale: Tensor[(_H, 1), config.dt],
    ) -> Tensor[(S, _H), config.dt]:
        # A dense SwiGLU every token goes through, scaled by the token's own
        # scalar gate. The gate is a projection to width one through a sigmoid,
        # so it is between 0 and 1 per token and cannot change sign.
        gate = tf.matmul(tokens, w_shared_gate)
        up = tf.matmul(tokens, w_shared_up)
        dense = tf.matmul(tf.mul(tf.mul(gate, tf.sigmoid(gate)), up), w_shared_down)
        scale = tf.sigmoid(tf.matmul(tokens, w_shared_scale))
        return tf.mul(dense, scale)

    @func
    def moe(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_post: Tensor[(_H,), config.dt],
        w_router: Tensor[(_H, _E), config.dt],
        w_gate: Tensor[(_E, _I, _H), config.dt],
        w_up: Tensor[(_E, _I, _H), config.dt],
        w_down: Tensor[(_E, _H, _I), config.dt],
        w_shared_gate: Tensor[(_H, _IS), config.dt],
        w_shared_up: Tensor[(_H, _IS), config.dt],
        w_shared_down: Tensor[(_IS, _H), config.dt],
        w_shared_scale: Tensor[(_H, 1), config.dt],
    ) -> Tensor[(1, S, _H), config.dt]:
        # Fused post_attention_layernorm + `Qwen3_5MoeSparseMoeBlock`, no
        # residual (the layer owns the residual add).
        tokens = tf.reshape(tf.rms_norm(hidden, gamma_post), new_shape=(S, _H))
        weights, indices = routing(tokens, w_router)
        routed = routed_experts(tokens, weights, indices, w_gate, w_up, w_down)
        shared = shared_expert(
            tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
        )
        return tf.reshape(tf.add(routed, shared), new_shape=(1, S, _H))
