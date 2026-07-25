"""Real-size DeepSeek V4 decode MoE HIR dataflow.

Two ``@module`` factories, matching this file's two real router variants:
``build_moe(config)`` (hash router, entry ``deepseek_v4_flash_moe_hash``,
real layers ``0..num_hash_layers-1``) and ``build_moe_learned(config)``
(learned/``noaux_tc`` router, entry ``deepseek_v4_flash_moe``, later real
layers; used by ``tests/schedule/*``).

Every shape is read directly off *config*, never a hardcoded literal, so the
same factory builds the real-scale modules below (``REAL``) and a tiny,
end-to-end-runnable one (``DSV4Config.tiny()``) for tests. The two
``where(layout=...)`` annotations inside each entry ``@func`` are an
exception: their shape positions parse only a literal or a bare ``Name``, no
attribute access, so each factory also binds a plain local (``dim``,
``n_act``) purely for those two spots.

Weights are derived automatically from every ``ConstTensor`` param. Routed
expert weights are fp8 e4m3 with a ``quant_block``-square ``ue8m0``
(``f8e8m0``) block scale; only the *scale* tensors need a converter (cast,
F32 on disk -> ``f8e8m0``).
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import REAL, DSV4Config
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

# Real-scale model constants, re-exported for callers that only need the
# numbers, not a built Module.
DIM = REAL.dim
N_ROUTED = REAL.n_routed
N_ACT = REAL.n_act
MOE_INTER = REAL.moe_inter
ROUTE_SCALE = REAL.route_scale
SWIGLU_LIMIT = REAL.swiglu_limit
VOCAB = REAL.vocab  # used by the hash router's tid2eid table below


def build_moe(config: DSV4Config) -> Module:
    """The hash-router MoE component (entry ``deepseek_v4_flash_moe_hash``,
    plus a plain ``forward``). Takes an already-normalized hidden state: the
    checkpoint's ``ffn_norm.weight`` is layer-level, so this component has no
    pre-MoE norm of its own (contrast ``build_moe_learned`` below).
    """
    dim = config.dim  # bare-Name locals for the two where(layout=...) spots
    n_act = config.n_act  # below only -- see the module docstring.

    @module(entry="deepseek_v4_flash_moe_hash")
    class DeepseekV4MoE:
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
            # Hash routing: expert ids come from a per-token-id table lookup
            # (tid2eid[token_ids]), not a learned top-k selection, and no bias
            # is added before the gather.
            #
            # tid2eid is stored i64 on disk despite being declared int32 in
            # the reference model -- loaded as i64 directly, matching
            # moe_experts_core's own eids parameter.
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
            # `hidden` arrives already normalized; see build_moe's docstring.
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
            """Hash-router MoE, end to end. ``hidden`` must already be
            normalized (see build_moe's docstring)."""
            return self.deepseek_v4_flash_moe_hash(hidden, token_ids)

    return DeepseekV4MoE


def build_moe_learned(config: DSV4Config) -> Module:
    """The learned/``noaux_tc``-router MoE component (entry
    ``deepseek_v4_flash_moe``). Used by ``tests/schedule/*``; keeps its own
    ``pre_moe_rms_norm`` (contrast ``build_moe`` above)."""
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
    """Lazily re-export the IR Function nodes for ``deepseek_v4_flash_moe`` /
    ``moe_experts_core`` / ``moe_topk`` (test_moe.py needs the node itself,
    for ``.params`` / ``.return_type`` / ``.body`` / ``.target``).

    Must stay lazy: binding these eagerly as plain module globals would make
    them resolve ahead of a factory's own class-body/factory-local lookup, so
    ``moe_topk``'s call to ``moe_experts_core(...)`` (and
    ``deepseek_v4_flash_moe``'s call to ``moe_topk(...)``) would keep
    resolving to this stale export instead of the fresh function a later
    ``build_moe_learned(config)`` call (e.g. at ``DSV4Config.tiny()``) built.
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
