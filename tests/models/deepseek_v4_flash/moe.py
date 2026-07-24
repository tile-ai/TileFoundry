"""Real-size DeepSeek V4 decode MoE HIR dataflow."""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

DIM = 4096
N_ROUTED = 256
N_ACT = 6
MOE_INTER = 2048
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
# DeepSeek-V3-tokenizer-sized vocabulary (config.json vocab_size) — shared by
# the hash router's ``tid2eid`` table below and by ``decode_step.py``'s
# embed/lm_head (which import it from here instead of duplicating it).
VOCAB = 129280


@func
def pre_moe_rms_norm(
    x: Tensor[(1, 1, DIM), "bf16"],
    weight: ConstTensor[(DIM,), "f32"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.rms_norm(x, weight)


@func
def shared_fp8_dequant_w1(
    weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
) -> Tensor[(MOE_INTER, DIM), "bf16"]:
    blocks = tf.reshape(
        tf.cast(weight, dtype="bf16"),
        new_shape=(MOE_INTER // 128, 128, DIM // 128, 128),
    )
    block_scale = tf.reshape(
        tf.cast(scale, dtype="bf16"),
        new_shape=(MOE_INTER // 128, 1, DIM // 128, 1),
    )
    return tf.reshape(blocks * block_scale, new_shape=(MOE_INTER, DIM))


@func
def shared_fp8_dequant_w2(
    weight: ConstTensor[(DIM, MOE_INTER), "fp8e4m3"],
    scale: ConstTensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(DIM, MOE_INTER), "bf16"]:
    blocks = tf.reshape(
        tf.cast(weight, dtype="bf16"),
        new_shape=(DIM // 128, 128, MOE_INTER // 128, 128),
    )
    block_scale = tf.reshape(
        tf.cast(scale, dtype="bf16"),
        new_shape=(DIM // 128, 1, MOE_INTER // 128, 1),
    )
    return tf.reshape(blocks * block_scale, new_shape=(DIM, MOE_INTER))


@func
def moe_experts_core(
    x: Tensor[(1, 1, DIM), "bf16"],
    gweights: Tensor[(1, N_ACT), "f32"],
    eids: Tensor[(1, N_ACT), "i64"],
    w1_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w1_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w3_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w3_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w2_weight: ConstTensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    w2_scale: ConstTensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
) -> Tensor[(1, N_ACT, DIM), "bf16"]:
    xt = tf.reshape(x, new_shape=(1, DIM))
    gathered_w1 = tf.cast(tf.gather(w1_weight, eids, axis=0), dtype="bf16")
    gathered_s1 = tf.cast(tf.gather(w1_scale, eids, axis=0), dtype="bf16")
    w1 = tf.reshape(
        tf.reshape(gathered_w1, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 32))
        * tf.reshape(gathered_s1, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 1)),
        new_shape=(1, N_ACT, MOE_INTER, DIM),
    )
    gathered_w3 = tf.cast(tf.gather(w3_weight, eids, axis=0), dtype="bf16")
    gathered_s3 = tf.cast(tf.gather(w3_scale, eids, axis=0), dtype="bf16")
    w3 = tf.reshape(
        tf.reshape(gathered_w3, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 32))
        * tf.reshape(gathered_s3, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 1)),
        new_shape=(1, N_ACT, MOE_INTER, DIM),
    )
    gathered_w2 = tf.cast(tf.gather(w2_weight, eids, axis=0), dtype="bf16")
    gathered_s2 = tf.cast(tf.gather(w2_scale, eids, axis=0), dtype="bf16")
    w2 = tf.reshape(
        tf.reshape(gathered_w2, new_shape=(1, N_ACT, DIM, MOE_INTER // 32, 32))
        * tf.reshape(gathered_s2, new_shape=(1, N_ACT, DIM, MOE_INTER // 32, 1)),
        new_shape=(1, N_ACT, DIM, MOE_INTER),
    )

    token = tf.reshape(xt, new_shape=(1, 1, DIM, 1))
    gate_value = tf.cast(
        tf.reshape(tf.matmul(w1, token), new_shape=(1, N_ACT, MOE_INTER)),
        dtype="f32",
    )
    up_value = tf.cast(
        tf.reshape(tf.matmul(w3, token), new_shape=(1, N_ACT, MOE_INTER)),
        dtype="f32",
    )
    limit = tf.full_like(up_value, value=SWIGLU_LIMIT)
    up_value = tf.maximum(
        tf.minimum(up_value, limit),
        tf.full_like(up_value, value=-SWIGLU_LIMIT),
    )
    gate_value = tf.minimum(gate_value, limit)
    hidden = (gate_value * tf.sigmoid(gate_value)) * up_value
    hidden = tf.reshape(
        tf.cast(hidden, dtype="bf16"),
        new_shape=(1, N_ACT, MOE_INTER, 1),
    )
    expert_output = tf.cast(
        tf.reshape(tf.matmul(w2, hidden), new_shape=(1, N_ACT, DIM)),
        dtype="f32",
    )
    weighted = expert_output * tf.reshape(
        gweights, new_shape=(1, N_ACT, 1)
    )
    return tf.cast(weighted, dtype="bf16")


@func
def moe_topk(
    x: Tensor[(1, 1, DIM), "bf16"],
    gate_weight: ConstTensor[(N_ROUTED, DIM), "bf16"],
    gate_bias: ConstTensor[(N_ROUTED,), "f32"],
    w1_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w1_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w3_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w3_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w2_weight: ConstTensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    w2_scale: ConstTensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
) -> Tensor[(1, N_ACT, DIM), "bf16"]:
    xt = tf.reshape(x, new_shape=(1, DIM))
    gate = tf.matmul(
        tf.cast(xt, dtype="f32"),
        tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
    )
    softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
    scores = softplus * tf.rsqrt(softplus)
    selection = scores + tf.reshape(gate_bias, new_shape=(1, N_ROUTED))
    _, eids = tf.topk(selection, k=N_ACT, axis=-1)
    gweights = tf.gather(scores, eids, axis=1, batch_dims=1)
    weight_sum = tf.reduce(
        gweights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM
    )
    gweights = (gweights / weight_sum) * tf.full_like(
        gweights, value=ROUTE_SCALE
    )
    return moe_experts_core(
        x,
        gweights,
        eids,
        w1_weight,
        w1_scale,
        w3_weight,
        w3_scale,
        w2_weight,
        w2_scale,
    )


@func
def moe_hash_gather(
    x: Tensor[(1, 1, DIM), "bf16"],
    gate_weight: ConstTensor[(N_ROUTED, DIM), "bf16"],
    tid2eid: ConstTensor[(VOCAB, N_ACT), "i64"],
    token_ids: Tensor[(1,), "i64"],
    w1_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w1_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w3_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w3_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w2_weight: ConstTensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    w2_scale: ConstTensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
) -> Tensor[(1, N_ACT, DIM), "bf16"]:
    # Hash routing (model.py Gate.forward, layer_id < config.json's
    # num_hash_layers==3): expert ids are a per-token-id table lookup
    # (tid2eid[input_ids]), not a learned top-k selection -- and (unlike the
    # learned/noaux_tc router in moe_topk) there is no bias added before the
    # gather: model.py's Gate.bias is None for a hash layer, so the gathered
    # routing weights come straight off the un-biased score_func output
    # ("original_scores" in model.py). score_func is config.json's
    # "sqrtsoftplus" for both routers, same scores formula as moe_topk, and
    # the post-gather normalize + route_scale tail is identical too (model.py
    # applies both unconditionally whenever score_func != "softmax", hash or
    # not) -- only the *selection* step (this function's ``tf.gather(tid2eid,
    # token_ids, ...)`` vs moe_topk's ``tf.topk``) differs. Routed-expert
    # weight/scale format (f4e2m1 + DIM//32-block f8e8m0) is the same real
    # checkpoint convention as moe_topk's, so moe_experts_core's dequant is
    # shared unmodified between both routers.
    #
    # Checkpoint quirk (confirmed empirically): tid2eid is declared
    # ``dtype=torch.int32`` in model.py but is actually stored **I64** on
    # disk in the real checkpoint -- loaded as i64 directly (no cast needed
    # here), which also happens to be exactly the dtype moe_experts_core's
    # own ``eids`` parameter requires.
    xt = tf.reshape(x, new_shape=(1, DIM))
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
        gweights, value=ROUTE_SCALE
    )
    return moe_experts_core(
        x,
        gweights,
        eids,
        w1_weight,
        w1_scale,
        w3_weight,
        w3_scale,
        w2_weight,
        w2_scale,
    )


@func
def shared_expert(
    x: Tensor[(1, 1, DIM), "bf16"],
    w1_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    w1_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    w3_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    w3_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    w2_weight: ConstTensor[(DIM, MOE_INTER), "fp8e4m3"],
    w2_scale: ConstTensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    xt = tf.reshape(x, new_shape=(1, DIM))
    w1 = shared_fp8_dequant_w1(w1_weight, w1_scale)
    w3 = shared_fp8_dequant_w1(w3_weight, w3_scale)
    gate = tf.cast(
        tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
    )
    up = tf.cast(
        tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32"
    )
    limit = tf.full_like(up, value=SWIGLU_LIMIT)
    up = tf.maximum(tf.minimum(up, limit), tf.full_like(up, value=-SWIGLU_LIMIT))
    gate = tf.minimum(gate, limit)
    hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
    w2 = shared_fp8_dequant_w2(w2_weight, w2_scale)
    output = tf.cast(
        tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16"
    )
    return tf.reshape(output, new_shape=(1, 1, DIM))


@func
def combine_expert_outputs(
    routed: Tensor[(1, 1, DIM), "bf16"],
    shared: Tensor[(1, 1, DIM), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.add(routed, shared)


@func(target=CudaTarget(), topologies=(Topology("cta", 132),))
def deepseek_v4_flash_moe(
    x: Tensor[(1, 1, DIM), "bf16"],
    rms_weight: ConstTensor[(DIM,), "f32"],
    gate_weight: ConstTensor[(N_ROUTED, DIM), "bf16"],
    gate_bias: ConstTensor[(N_ROUTED,), "f32"],
    routed_w1_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w1_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w3_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w3_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w2_weight: ConstTensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    routed_w2_scale: ConstTensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
    shared_w1_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w1_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w3_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w3_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w2_weight: ConstTensor[(DIM, MOE_INTER), "fp8e4m3"],
    shared_w2_scale: ConstTensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    hidden = pre_moe_rms_norm(x, rms_weight)
    routed_experts: where(layout=(_, 6 @ cta, DIM)) = moe_topk(
        hidden,
        gate_weight,
        gate_bias,
        routed_w1_weight,
        routed_w1_scale,
        routed_w3_weight,
        routed_w3_scale,
        routed_w2_weight,
        routed_w2_scale,
    )
    routed_reduced = tf.reduce(
        routed_experts,
        axes=(1,),
        keepdim=False,
        kind=ReduceKind.SUM,
    )
    routed_value = tf.reshape(
        tf.cast(routed_reduced, dtype="bf16"),
        new_shape=(1, 1, DIM),
    )
    shared_value = shared_expert(
        hidden,
        shared_w1_weight,
        shared_w1_scale,
        shared_w3_weight,
        shared_w3_scale,
        shared_w2_weight,
        shared_w2_scale,
    )
    combined: where(layout=((_, _, DIM), {cta @ B()})) = combine_expert_outputs(
        routed_value,
        shared_value,
    )
    return combined


@func(target=CudaTarget(), topologies=(Topology("cta", 132),))
def deepseek_v4_flash_moe_hash(
    x: Tensor[(1, 1, DIM), "bf16"],
    rms_weight: ConstTensor[(DIM,), "f32"],
    gate_weight: ConstTensor[(N_ROUTED, DIM), "bf16"],
    tid2eid: ConstTensor[(VOCAB, N_ACT), "i64"],
    token_ids: Tensor[(1,), "i64"],
    routed_w1_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w1_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w3_weight: ConstTensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w3_scale: ConstTensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w2_weight: ConstTensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    routed_w2_scale: ConstTensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
    shared_w1_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w1_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w3_weight: ConstTensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w3_scale: ConstTensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w2_weight: ConstTensor[(DIM, MOE_INTER), "fp8e4m3"],
    shared_w2_scale: ConstTensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    # Hash-router twin of deepseek_v4_flash_moe (config.json's first
    # num_hash_layers==3 real layers) -- identical reduce/combine tail and
    # identical weight/scale formats throughout; only the routed-expert
    # selection call differs (moe_hash_gather vs moe_topk; see
    # moe_hash_gather's docstring). deepseek_v4_flash_moe (moe_topk) covers
    # real layers >= num_hash_layers==3; this one covers real layers 0..2.
    hidden = pre_moe_rms_norm(x, rms_weight)
    routed_experts: where(layout=(_, 6 @ cta, DIM)) = moe_hash_gather(
        hidden,
        gate_weight,
        tid2eid,
        token_ids,
        routed_w1_weight,
        routed_w1_scale,
        routed_w3_weight,
        routed_w3_scale,
        routed_w2_weight,
        routed_w2_scale,
    )
    routed_reduced = tf.reduce(
        routed_experts,
        axes=(1,),
        keepdim=False,
        kind=ReduceKind.SUM,
    )
    routed_value = tf.reshape(
        tf.cast(routed_reduced, dtype="bf16"),
        new_shape=(1, 1, DIM),
    )
    shared_value = shared_expert(
        hidden,
        shared_w1_weight,
        shared_w1_scale,
        shared_w3_weight,
        shared_w3_scale,
        shared_w2_weight,
        shared_w2_scale,
    )
    combined: where(layout=((_, _, DIM), {cta @ B()})) = combine_expert_outputs(
        routed_value,
        shared_value,
    )
    return combined


deepseek_v4_flash_module = Module(
    name="DeepSeekV4FlashMoe",
    functions=(
        shared_fp8_dequant_w1,
        shared_fp8_dequant_w2,
        pre_moe_rms_norm,
        moe_experts_core,
        moe_topk,
        shared_expert,
        combine_expert_outputs,
        deepseek_v4_flash_moe,
    ),
    entry="deepseek_v4_flash_moe",
)

moe_hash_module = Module(
    name="moe",
    functions=(
        pre_moe_rms_norm,
        moe_experts_core,
        moe_hash_gather,
        combine_expert_outputs,
        deepseek_v4_flash_moe_hash,
        shared_fp8_dequant_w1,
        shared_fp8_dequant_w2,
        shared_expert,
    ),
    entry="deepseek_v4_flash_moe_hash",
)


__all__ = [
    "DIM",
    "MOE_INTER",
    "N_ACT",
    "N_ROUTED",
    "ROUTE_SCALE",
    "SWIGLU_LIMIT",
    "VOCAB",
    "combine_expert_outputs",
    "deepseek_v4_flash_moe",
    "deepseek_v4_flash_moe_hash",
    "deepseek_v4_flash_module",
    "moe_experts_core",
    "moe_hash_gather",
    "moe_hash_module",
    "moe_topk",
    "pre_moe_rms_norm",
    "shared_expert",
    "shared_fp8_dequant_w1",
    "shared_fp8_dequant_w2",
]
