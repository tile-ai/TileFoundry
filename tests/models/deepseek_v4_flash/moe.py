"""Real-size DeepSeek V4 decode MoE HIR dataflow.

Two ``@module`` factories, matching this file's two real router variants
(config.json: real layers ``0..num_hash_layers-1`` (``==3``) hash-route;
every later layer uses the learned/``noaux_tc`` router). Each factory's inner
class is a CapWords, HF-style *type* name (``DeepseekV4MoE``,
``DeepseekV4NoauxTcMoE``) -- matching ``attention.py``'s
``build_attention(config) -> class DeepseekV4Attention:`` shape. The node's
actual name in a tree comes from whatever attribute a parent nests it under
(``@module``'s own torch/HF semantics: ``self.moe = build_moe(config)``
renames the child to ``"moe"`` the same way ``self.self_attn =
DeepseekV4Attention(config)`` renames that one to ``"self_attn"``), which is
what makes the checkpoint-alias entry (``"moe" -> "ffn"``) resolve once a
decoder layer nests this under a ``moe`` attribute -- the class name itself
is cosmetic, not load-bearing for addressing.

- ``build_moe(config)`` -- the hash-router component (entry
  ``deepseek_v4_flash_moe_hash``), plus a plain ``forward``. This is the
  component ``causal_lm.py``'s decoder layer nests under its ``moe``
  attribute.
- ``build_moe_learned(config)`` -- the learned-router variant (entry
  ``deepseek_v4_flash_moe``), used by ``tests/schedule/*``.

Pre-MoE norm asymmetry between the two factories (deliberate, not an
oversight): the real checkpoint's per-layer ``ffn_norm.weight`` is a
LAYER-level tensor (a sibling of ``ffn``, not one of its own tensors), so the
textbook-/checkpoint-faithful arrangement is for the *layer* to normalize
before calling the MoE component, not the component itself.
``build_moe``'s ``deepseek_v4_flash_moe_hash`` follows that: it takes an
already-normalized hidden state and has no pre-MoE norm of its own (see
``causal_lm.py``, its only caller, which owns ``ffn_norm.weight`` under its
own ``pre_moe_rms_norm``). ``build_moe_learned`` still runs its own internal
``pre_moe_rms_norm`` against a component-local ``rms_weight`` that has no
real checkpoint tensor backing it -- left as is, on purpose: that root is
only ever built standalone for ``tests/schedule/*`` planning/scheduling
fixtures (never nested under a real decoder layer), and ``test_moe.py``
asserts on ``deepseek_v4_flash_moe.params[4]`` / ``params[8]`` *by index* --
dropping a leading param there would both break those frozen assertions and
reshape the planning problem itself. If ``build_moe_learned`` ever gets
nested under a real layer the same way, drop its pre-norm too, matching
``build_moe``.

Every shape is read directly off *config* (``config.dim``, ``config.moe_inter``,
``config.n_routed``, ``config.n_act``, ``config.vocab``, ``config.route_scale``,
``config.swiglu_limit``, and ``config.blocks(extent)`` for a weight's
``quant_block``-square block-scale grid) -- never a hardcoded literal -- so
the same factory builds the real-scale modules below (``REAL``) and a tiny,
end-to-end-runnable one (``DSV4Config.tiny()``) for tests. A ``@func``
defined inside a factory closes over the factory's own ``config`` parameter
directly in its annotations: ``_definition_namespace`` (``tilefoundry.
script``) walks the enclosing lexical scopes outward -- the ``@module``
class body first, then the factory's own locals (where ``config`` lives), then
the module -- and merges them *below* the function's own globals/freevars,
so this can only add names, never shadow one. This is the standard style for
every model fixture now.

The two ``where(layout=...)`` annotations inside each entry ``@func`` are the
one exception: their shape positions parse only a literal or a bare ``Name``
-- no attribute access (see ``tilefoundry.parser.hir_parser.
_resolve_layout_extent``) -- so each factory also binds a plain local
(``dim = config.dim``, ``n_act = config.n_act``) purely for those two spots; the
same outward-scope walk makes them visible there too.

Weights are derived automatically from every ``ConstTensor`` param -- never
declared twice, no ``weights=`` / ``states=`` anywhere
(``tilefoundry.module``). The real checkpoint (``DeepSeek-V4-Flash-FP8``)
quantizes every routed/shared expert weight fp8 e4m3 with a
``quant_block``-square (128x128 at real scale) ``ue8m0`` (``f8e8m0``) block
scale (config.json's ``quantization_config``); the routed experts' 256
per-expert tensors are named by a one-to-many alias and ``prepare`` stacks
them along a new leading axis before anything else happens, so that stacked
form already counts as the weight's raw/canonical form -- only the *scale*
tensors need a converter (a cast, F32 on disk -> ``f8e8m0``), registered with
``<owner>.converter("<name>")``.
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import REAL, DSV4Config
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

# Real-scale model constants (config.json, via DSV4Config) -- re-exported so
# callers/tests that only need the numbers (not a built Module) keep working
# unchanged. Fixed at REAL scale permanently: build_moe / build_moe_learned's
# own @funcs read every shape off their *config* parameter directly (see the
# module docstring), never off these, so calling either factory again later
# (e.g. at DSV4Config.tiny()) never touches them.
DIM = REAL.dim
N_ROUTED = REAL.n_routed
N_ACT = REAL.n_act
MOE_INTER = REAL.moe_inter
ROUTE_SCALE = REAL.route_scale
SWIGLU_LIMIT = REAL.swiglu_limit
# DeepSeek-V3-tokenizer-sized vocabulary (config.json vocab_size) -- used by
# the hash router's ``tid2eid`` table below; re-exported at module level (like
# the other constants above) for any external caller that wants the number
# without a Module. ``causal_lm.py``'s own ``embed``/``lm_head`` read
# ``config.vocab`` directly off their own ``config: DSV4Config`` parameter
# instead (config-driven, like everything else there), rather than importing
# this constant.
VOCAB = REAL.vocab


def build_moe(config: DSV4Config) -> Module:
    """The hash-router MoE component (real layers ``0..num_hash_layers-1``,
    ``==3`` per config.json): entry ``deepseek_v4_flash_moe_hash``, plus a
    plain ``forward`` -- the component the end-to-end test nests under a
    decoder layer. Takes an already-normalized hidden state: the checkpoint's
    ``ffn_norm.weight`` is a layer-level tensor (a sibling of ``ffn``, not
    inside it), so the layer normalizes and this component has no pre-MoE
    norm of its own (contrast ``build_moe_learned`` below)."""
    dim = config.dim  # bare-Name locals for the two where(layout=...) spots
    n_act = config.n_act  # below only -- see the module docstring.

    @module(entry="deepseek_v4_flash_moe_hash")
    class DeepseekV4MoE:
        # No internal pre-MoE RMSNorm here (unlike build_moe_learned below):
        # the real checkpoint's per-layer `ffn_norm.weight` is a LAYER-level
        # tensor (a sibling of `ffn`, not inside it, per config.json's
        # checkpoint layout), so the layer owns that norm and this
        # component's fused entry (deepseek_v4_flash_moe_hash, below) takes
        # an already-normalized hidden state instead of normalizing its own
        # (previously fabricated, checkpoint-less) copy -- see its docstring.
        @func
        def shared_fp8_dequant_w1(
            weight: Tensor[(config.moe_inter, config.dim), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"],
        ) -> Tensor[(config.moe_inter, config.dim), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.moe_inter), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.moe_inter, config.dim))

        @func
        def shared_fp8_dequant_w2(
            weight: Tensor[(config.dim, config.moe_inter), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"],
        ) -> Tensor[(config.dim, config.moe_inter), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.dim), config.quant_block,
                    config.blocks(config.moe_inter), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.dim, config.moe_inter))

        @func
        def moe_experts_core(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gweights: Tensor[(1, config.n_act), "f32"],
            eids: Tensor[(1, config.n_act), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))

            gathered_w1 = tf.cast(tf.gather(w1_weight, eids, axis=0), dtype="bf16")
            gathered_s1 = tf.cast(tf.gather(w1_scale, eids, axis=0), dtype="bf16")
            w1 = tf.reshape(
                tf.reshape(
                    gathered_w1,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s1,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w3 = tf.cast(tf.gather(w3_weight, eids, axis=0), dtype="bf16")
            gathered_s3 = tf.cast(tf.gather(w3_scale, eids, axis=0), dtype="bf16")
            w3 = tf.reshape(
                tf.reshape(
                    gathered_w3,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s3,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w2 = tf.cast(tf.gather(w2_weight, eids, axis=0), dtype="bf16")
            gathered_s2 = tf.cast(tf.gather(w2_scale, eids, axis=0), dtype="bf16")
            w2 = tf.reshape(
                tf.reshape(
                    gathered_w2,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.dim), config.quant_block,
                        config.blocks(config.moe_inter), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s2,
                    new_shape=(
                        1, config.n_act, config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.dim, config.moe_inter),
            )

            token = tf.reshape(xt, new_shape=(1, 1, config.dim, 1))
            gate_value = tf.cast(
                tf.reshape(tf.matmul(w1, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            up_value = tf.cast(
                tf.reshape(tf.matmul(w3, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            limit = tf.full_like(up_value, value=config.swiglu_limit)
            up_value = tf.maximum(
                tf.minimum(up_value, limit),
                tf.full_like(up_value, value=-config.swiglu_limit),
            )
            gate_value = tf.minimum(gate_value, limit)
            hidden = (gate_value * tf.sigmoid(gate_value)) * up_value
            hidden = tf.reshape(
                tf.cast(hidden, dtype="bf16"),
                new_shape=(1, config.n_act, config.moe_inter, 1),
            )
            expert_output = tf.cast(
                tf.reshape(tf.matmul(w2, hidden), new_shape=(1, config.n_act, config.dim)),
                dtype="f32",
            )
            weighted = expert_output * tf.reshape(gweights, new_shape=(1, config.n_act, 1))
            return tf.cast(weighted, dtype="bf16")

        @moe_experts_core.converter("w1_scale")
        def _(
            w1_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w1_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w3_scale")
        def _(
            w3_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w3_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w2_scale")
        def _(
            w2_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(w2_scale_raw, dtype="f8e8m0")

        @func
        def moe_hash_gather(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            tid2eid: ConstTensor[(config.vocab, config.n_act), "i64"],
            token_ids: Tensor[(1,), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            # Hash routing (model.py Gate.forward, layer_id < config.json's
            # num_hash_layers==3): expert ids are a per-token-id table
            # lookup (tid2eid[input_ids]), not a learned top-k selection --
            # and (unlike the learned/noaux_tc router in moe_topk) there is
            # no bias added before the gather: model.py's Gate.bias is None
            # for a hash layer, so the gathered routing weights come
            # straight off the un-biased score_func output
            # ("original_scores" in model.py). score_func is config.json's
            # "sqrtsoftplus" for both routers, same scores formula as
            # moe_topk, and the post-gather normalize + route_scale tail is
            # identical too (model.py applies both unconditionally whenever
            # score_func != "softmax", hash or not) -- only the *selection*
            # step (this function's ``tf.gather(tid2eid, token_ids, ...)``
            # vs moe_topk's ``tf.topk``) differs. Routed-expert weight/scale
            # format (fp8 e4m3 + a quant_block-square ue8m0 block-scale grid
            # per expert) is the same real checkpoint convention as
            # moe_topk's, so moe_experts_core's dequant is shared unmodified
            # between both routers.
            #
            # Checkpoint quirk (confirmed empirically): tid2eid is declared
            # ``dtype=torch.int32`` in model.py but is actually stored
            # **I64** on disk in the real checkpoint -- loaded as i64
            # directly (no cast needed here), which also happens to be
            # exactly the dtype moe_experts_core's own ``eids`` parameter
            # requires.
            xt = tf.reshape(x, new_shape=(1, config.dim))
            gate = tf.matmul(
                tf.cast(xt, dtype="f32"),
                tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
            )
            softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
            scores = softplus * tf.rsqrt(softplus)
            eids = tf.gather(tid2eid, token_ids, axis=0)
            gweights = tf.gather(scores, eids, axis=1, batch_dims=1)
            weight_sum = tf.reduce(
                gweights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM
            )
            gweights = (gweights / weight_sum) * tf.full_like(
                gweights, value=config.route_scale
            )
            return moe_experts_core(
                x, gweights, eids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )

        @func
        def shared_expert(
            x: Tensor[(1, 1, config.dim), "bf16"],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            w1 = shared_fp8_dequant_w1(shared_w1_weight, shared_w1_scale)
            w3 = shared_fp8_dequant_w1(shared_w3_weight, shared_w3_scale)
            gate = tf.cast(
                tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
            )
            up = tf.cast(
                tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32"
            )
            limit = tf.full_like(up, value=config.swiglu_limit)
            up = tf.maximum(
                tf.minimum(up, limit), tf.full_like(up, value=-config.swiglu_limit)
            )
            gate = tf.minimum(gate, limit)
            hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
            w2 = shared_fp8_dequant_w2(shared_w2_weight, shared_w2_scale)
            output = tf.cast(
                tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16"
            )
            return tf.reshape(output, new_shape=(1, 1, config.dim))

        @shared_expert.converter("shared_w1_scale")
        def _(
            shared_w1_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w1_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w3_scale")
        def _(
            shared_w3_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w3_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w2_scale")
        def _(
            shared_w2_scale_raw: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(shared_w2_scale_raw, dtype="f8e8m0")

        @func
        def combine_expert_outputs(
            routed: Tensor[(1, 1, config.dim), "bf16"],
            shared: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.add(routed, shared)

        @func(target=CudaTarget(), topologies=(Topology("cta", 132),))
        def deepseek_v4_flash_moe_hash(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            tid2eid: ConstTensor[(config.vocab, config.n_act), "i64"],
            token_ids: Tensor[(1,), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            # Hash-router twin of deepseek_v4_flash_moe (config.json's
            # first num_hash_layers==3 real layers) -- identical
            # reduce/combine tail and identical weight/scale formats
            # throughout; only the routed-expert selection call differs
            # (moe_hash_gather vs moe_topk; see moe_hash_gather's
            # docstring). deepseek_v4_flash_moe (moe_topk) covers real
            # layers >= num_hash_layers==3; this one covers real layers
            # 0..2.
            #
            # `hidden` arrives already normalized: the real checkpoint keeps
            # `ffn_norm.weight` at the layer level (a sibling of `ffn`, not a
            # tensor inside it), so the layer computes `ffn_norm(h)` itself
            # and calls this component with the result -- there is no
            # component-local pre-MoE norm here (contrast build_moe_learned,
            # whose root is planning-only and keeps its own for now; see the
            # module docstring's asymmetry note).
            routed_experts: where(layout=(_, n_act @ cta, dim)) = moe_hash_gather(
                hidden, gate_weight, tid2eid, token_ids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )
            routed_reduced = tf.reduce(
                routed_experts, axes=(1,), keepdim=False, kind=ReduceKind.SUM,
            )
            routed_value = tf.reshape(
                tf.cast(routed_reduced, dtype="bf16"), new_shape=(1, 1, config.dim),
            )
            shared_value = shared_expert(
                hidden, shared_w1_weight, shared_w1_scale,
                shared_w3_weight, shared_w3_scale, shared_w2_weight, shared_w2_scale,
            )
            combined: where(layout=((_, _, dim), {cta @ B()})) = combine_expert_outputs(
                routed_value, shared_value,
            )
            return combined

        def forward(self, hidden, token_ids):
            """Hash-router MoE component, end to end (route/gather ->
            experts -> shared expert -> combine, all inside the fused entry
            @func above): weights are filled by name from ``self._bound``
            (``Module.load``); only activations are parameters here.
            ``hidden`` must already be normalized -- this component has no
            pre-MoE norm of its own (the caller's `ffn_norm.weight` is a
            layer-level checkpoint tensor; see the module docstring)."""
            return self.deepseek_v4_flash_moe_hash(hidden, token_ids)

    return DeepseekV4MoE


def build_moe_learned(config: DSV4Config) -> Module:
    """The learned/``noaux_tc``-router MoE component (real layers
    ``>= num_hash_layers``, i.e. ``>= 3`` per config.json): entry
    ``deepseek_v4_flash_moe``. Used by ``tests/schedule/*`` (structural /
    scheduling fixtures only)."""
    dim = config.dim  # bare-Name locals for the two where(layout=...) spots
    n_act = config.n_act  # below only -- see the module docstring.

    @module(entry="deepseek_v4_flash_moe")
    class DeepseekV4NoauxTcMoE:
        @func
        def pre_moe_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            rms_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, rms_weight)

        @func
        def shared_fp8_dequant_w1(
            weight: Tensor[(config.moe_inter, config.dim), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"],
        ) -> Tensor[(config.moe_inter, config.dim), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.moe_inter), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.moe_inter, config.dim))

        @func
        def shared_fp8_dequant_w2(
            weight: Tensor[(config.dim, config.moe_inter), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"],
        ) -> Tensor[(config.dim, config.moe_inter), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.dim), config.quant_block,
                    config.blocks(config.moe_inter), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.dim, config.moe_inter))

        @func
        def moe_experts_core(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gweights: Tensor[(1, config.n_act), "f32"],
            eids: Tensor[(1, config.n_act), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))

            gathered_w1 = tf.cast(tf.gather(w1_weight, eids, axis=0), dtype="bf16")
            gathered_s1 = tf.cast(tf.gather(w1_scale, eids, axis=0), dtype="bf16")
            w1 = tf.reshape(
                tf.reshape(
                    gathered_w1,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s1,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w3 = tf.cast(tf.gather(w3_weight, eids, axis=0), dtype="bf16")
            gathered_s3 = tf.cast(tf.gather(w3_scale, eids, axis=0), dtype="bf16")
            w3 = tf.reshape(
                tf.reshape(
                    gathered_w3,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s3,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w2 = tf.cast(tf.gather(w2_weight, eids, axis=0), dtype="bf16")
            gathered_s2 = tf.cast(tf.gather(w2_scale, eids, axis=0), dtype="bf16")
            w2 = tf.reshape(
                tf.reshape(
                    gathered_w2,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.dim), config.quant_block,
                        config.blocks(config.moe_inter), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s2,
                    new_shape=(
                        1, config.n_act, config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.dim, config.moe_inter),
            )

            token = tf.reshape(xt, new_shape=(1, 1, config.dim, 1))
            gate_value = tf.cast(
                tf.reshape(tf.matmul(w1, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            up_value = tf.cast(
                tf.reshape(tf.matmul(w3, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            limit = tf.full_like(up_value, value=config.swiglu_limit)
            up_value = tf.maximum(
                tf.minimum(up_value, limit),
                tf.full_like(up_value, value=-config.swiglu_limit),
            )
            gate_value = tf.minimum(gate_value, limit)
            hidden = (gate_value * tf.sigmoid(gate_value)) * up_value
            hidden = tf.reshape(
                tf.cast(hidden, dtype="bf16"),
                new_shape=(1, config.n_act, config.moe_inter, 1),
            )
            expert_output = tf.cast(
                tf.reshape(tf.matmul(w2, hidden), new_shape=(1, config.n_act, config.dim)),
                dtype="f32",
            )
            weighted = expert_output * tf.reshape(gweights, new_shape=(1, config.n_act, 1))
            return tf.cast(weighted, dtype="bf16")

        @moe_experts_core.converter("w1_scale")
        def _(
            w1_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w1_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w3_scale")
        def _(
            w3_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w3_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w2_scale")
        def _(
            w2_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(w2_scale_raw, dtype="f8e8m0")

        @func
        def moe_topk(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            gate_bias: ConstTensor[(config.n_routed,), "f32"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            gate = tf.matmul(
                tf.cast(xt, dtype="f32"),
                tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
            )
            softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
            scores = softplus * tf.rsqrt(softplus)
            selection = scores + tf.reshape(gate_bias, new_shape=(1, config.n_routed))
            _, eids = tf.topk(selection, k=config.n_act, axis=-1)
            gweights = tf.gather(scores, eids, axis=1, batch_dims=1)
            weight_sum = tf.reduce(
                gweights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM
            )
            gweights = (gweights / weight_sum) * tf.full_like(
                gweights, value=config.route_scale
            )
            return moe_experts_core(
                x, gweights, eids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )

        @func
        def shared_expert(
            x: Tensor[(1, 1, config.dim), "bf16"],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            w1 = shared_fp8_dequant_w1(shared_w1_weight, shared_w1_scale)
            w3 = shared_fp8_dequant_w1(shared_w3_weight, shared_w3_scale)
            gate = tf.cast(
                tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
            )
            up = tf.cast(
                tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32"
            )
            limit = tf.full_like(up, value=config.swiglu_limit)
            up = tf.maximum(
                tf.minimum(up, limit), tf.full_like(up, value=-config.swiglu_limit)
            )
            gate = tf.minimum(gate, limit)
            hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
            w2 = shared_fp8_dequant_w2(shared_w2_weight, shared_w2_scale)
            output = tf.cast(
                tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16"
            )
            return tf.reshape(output, new_shape=(1, 1, config.dim))

        @shared_expert.converter("shared_w1_scale")
        def _(
            shared_w1_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w1_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w3_scale")
        def _(
            shared_w3_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w3_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w2_scale")
        def _(
            shared_w2_scale_raw: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(shared_w2_scale_raw, dtype="f8e8m0")

        @func
        def combine_expert_outputs(
            routed: Tensor[(1, 1, config.dim), "bf16"],
            shared: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.add(routed, shared)

        @func(target=CudaTarget(), topologies=(Topology("cta", 132),))
        def deepseek_v4_flash_moe(
            x: Tensor[(1, 1, config.dim), "bf16"],
            rms_weight: ConstTensor[(config.dim,), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            gate_bias: ConstTensor[(config.n_routed,), "f32"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            hidden = pre_moe_rms_norm(x, rms_weight)
            routed_experts: where(layout=(_, n_act @ cta, dim)) = moe_topk(
                hidden, gate_weight, gate_bias,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )
            routed_reduced = tf.reduce(
                routed_experts, axes=(1,), keepdim=False, kind=ReduceKind.SUM,
            )
            routed_value = tf.reshape(
                tf.cast(routed_reduced, dtype="bf16"), new_shape=(1, 1, config.dim),
            )
            shared_value = shared_expert(
                hidden, shared_w1_weight, shared_w1_scale,
                shared_w3_weight, shared_w3_scale, shared_w2_weight, shared_w2_scale,
            )
            combined: where(layout=((_, _, dim), {cta @ B()})) = combine_expert_outputs(
                routed_value, shared_value,
            )
            return combined

    return DeepseekV4NoauxTcMoE


moe_hash_module = build_moe(REAL)
deepseek_v4_flash_module = build_moe_learned(REAL)

_LAZY_FUNCTIONS = ("deepseek_v4_flash_moe", "moe_experts_core", "moe_topk")


def __getattr__(name: str):
    """Individual @funcs, re-exported for external imports (test_moe.py's
    ``deepseek_v4_flash_moe`` / ``moe_experts_core`` / ``moe_topk``) by their
    IR Function node (``lookup``, not attribute access -- ``Module.<name>``
    now returns a *callable* that runs the function with weights auto-filled;
    the tests need the node itself, for ``.params`` / ``.return_type`` /
    ``.body`` / ``.target``).

    Resolved lazily (PEP 562 module ``__getattr__``, the same pattern
    ``tilefoundry.dsl``'s ``tf`` / ``T`` op namespaces use) rather than bound
    as a plain module attribute at import time: a real module-global of the
    same bare name would sit *above* a factory's own class-body/factory-local
    sibling lookup (``_definition_namespace`` merges its outward-scope walk
    *below* the function's own globals/freevars, so it can only add names,
    never shadow one), so binding it eagerly would make ``moe_topk``'s own
    call to ``moe_experts_core(...)`` -- and ``deepseek_v4_flash_moe``'s own
    call to ``moe_topk(...)`` -- resolve to this stale, already-exported
    (real-scale) function instead of the fresh one *any later*
    ``build_moe_learned(config)`` call (e.g. at ``DSV4Config.tiny()``) just
    built, breaking that rebuild. Not applied to ``deepseek_v4_flash_module``
    / ``moe_hash_module`` themselves (plain module attributes, nothing
    internally calls a *Module* by a bare name) or to ``build_moe``'s own
    entry (``deepseek_v4_flash_moe_hash``, not re-exported at all -- nothing
    outside this file needs it by name; ``moe_hash_module`` is used whole).
    """
    if name in _LAZY_FUNCTIONS:
        return deepseek_v4_flash_module.lookup(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DIM",
    "MOE_INTER",
    "N_ACT",
    "N_ROUTED",
    "ROUTE_SCALE",
    "SWIGLU_LIMIT",
    "VOCAB",
    "build_moe",
    "build_moe_learned",
    "deepseek_v4_flash_moe",  # noqa: F822 -- resolved by __getattr__ above, not a plain assignment
    "deepseek_v4_flash_module",
    "moe_experts_core",  # noqa: F822 -- resolved by __getattr__ above, not a plain assignment
    "moe_hash_module",
    "moe_topk",  # noqa: F822 -- resolved by __getattr__ above, not a plain assignment
]
