"""DeepSeek V4 decode MoE-layer HIR model.

The dimensions and routing math match the DSV4 decode reference. The layer is
written with ordinary HIR functions so later distribution tests can inspect the
pre-MoE normalization, routed experts, shared expert, and final combination as
separate call boundaries.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import ReduceKind, Tensor, tf

DIM = 4096
N_ROUTED = 256
N_ACT = 6
MOE_INTER = 2048
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0


@func
def pre_moe_rms_norm(
    x: Tensor[(1, 1, DIM), "bf16"],
    weight: Tensor[(DIM,), "f32"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.rms_norm(x, weight)


@func
def _dequant_shared_w1(
    weight: Tensor[(MOE_INTER, DIM), "fp8e4m3"],
    scale: Tensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
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
def _dequant_shared_w2(
    weight: Tensor[(DIM, MOE_INTER), "fp8e4m3"],
    scale: Tensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
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
def routed_expert(
    x: Tensor[(1, 1, DIM), "bf16"],
    gate_weight: Tensor[(N_ROUTED, DIM), "bf16"],
    gate_bias: Tensor[(N_ROUTED,), "f32"],
    w1_weight: Tensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w1_scale: Tensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w3_weight: Tensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    w3_scale: Tensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    w2_weight: Tensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    w2_scale: Tensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    xt = tf.reshape(x, new_shape=(1, DIM))
    gate = tf.matmul(
        tf.cast(xt, dtype="f32"),
        tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
    )
    softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
    scores = softplus * tf.rsqrt(softplus)
    selection = scores + tf.reshape(gate_bias, new_shape=(1, N_ROUTED))
    _, expert_ids = tf.topk(selection, k=N_ACT, axis=-1)
    weights = tf.gather(scores, expert_ids, axis=1, batch_dims=1)
    weight_sum = tf.reduce(weights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    weights = (weights / weight_sum) * tf.full_like(weights, value=ROUTE_SCALE)

    gathered_w1 = tf.cast(tf.gather(w1_weight, expert_ids, axis=0), dtype="bf16")
    gathered_s1 = tf.cast(tf.gather(w1_scale, expert_ids, axis=0), dtype="bf16")
    w1 = tf.reshape(
        tf.reshape(gathered_w1, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 32))
        * tf.reshape(gathered_s1, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 1)),
        new_shape=(1, N_ACT, MOE_INTER, DIM),
    )
    gathered_w3 = tf.cast(tf.gather(w3_weight, expert_ids, axis=0), dtype="bf16")
    gathered_s3 = tf.cast(tf.gather(w3_scale, expert_ids, axis=0), dtype="bf16")
    w3 = tf.reshape(
        tf.reshape(gathered_w3, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 32))
        * tf.reshape(gathered_s3, new_shape=(1, N_ACT, MOE_INTER, DIM // 32, 1)),
        new_shape=(1, N_ACT, MOE_INTER, DIM),
    )
    gathered_w2 = tf.cast(tf.gather(w2_weight, expert_ids, axis=0), dtype="bf16")
    gathered_s2 = tf.cast(tf.gather(w2_scale, expert_ids, axis=0), dtype="bf16")
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
    hidden = tf.reshape(tf.cast(hidden, dtype="bf16"), new_shape=(1, N_ACT, MOE_INTER, 1))
    expert_output = tf.cast(
        tf.reshape(tf.matmul(w2, hidden), new_shape=(1, N_ACT, DIM)),
        dtype="f32",
    )
    routed = tf.reduce(
        expert_output * tf.reshape(weights, new_shape=(1, N_ACT, 1)),
        axes=(1,),
        keepdim=False,
        kind=ReduceKind.SUM,
    )
    return tf.reshape(tf.cast(routed, dtype="bf16"), new_shape=(1, 1, DIM))


@func
def shared_expert(
    x: Tensor[(1, 1, DIM), "bf16"],
    w1_weight: Tensor[(MOE_INTER, DIM), "fp8e4m3"],
    w1_scale: Tensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    w3_weight: Tensor[(MOE_INTER, DIM), "fp8e4m3"],
    w3_scale: Tensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    w2_weight: Tensor[(DIM, MOE_INTER), "fp8e4m3"],
    w2_scale: Tensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    xt = tf.reshape(x, new_shape=(1, DIM))
    w1 = _dequant_shared_w1(w1_weight, w1_scale)
    w3 = _dequant_shared_w1(w3_weight, w3_scale)
    gate = tf.cast(
        tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
    )
    up = tf.cast(tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32")
    limit = tf.full_like(up, value=SWIGLU_LIMIT)
    up = tf.maximum(tf.minimum(up, limit), tf.full_like(up, value=-SWIGLU_LIMIT))
    gate = tf.minimum(gate, limit)
    hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
    w2 = _dequant_shared_w2(w2_weight, w2_scale)
    output = tf.cast(tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16")
    return tf.reshape(output, new_shape=(1, 1, DIM))


@func
def combine_expert_outputs(
    routed: Tensor[(1, 1, DIM), "bf16"],
    shared: Tensor[(1, 1, DIM), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.add(routed, shared)


def dsv4_moe_layer(
    x: Tensor[(1, 1, DIM), "bf16"],
    rms_weight: Tensor[(DIM,), "f32"],
    gate_weight: Tensor[(N_ROUTED, DIM), "bf16"],
    gate_bias: Tensor[(N_ROUTED,), "f32"],
    routed_w1_weight: Tensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w1_scale: Tensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w3_weight: Tensor[(N_ROUTED, MOE_INTER, DIM), "f4e2m1"],
    routed_w3_scale: Tensor[(N_ROUTED, MOE_INTER, DIM // 32), "f8e8m0"],
    routed_w2_weight: Tensor[(N_ROUTED, DIM, MOE_INTER), "f4e2m1"],
    routed_w2_scale: Tensor[(N_ROUTED, DIM, MOE_INTER // 32), "f8e8m0"],
    shared_w1_weight: Tensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w1_scale: Tensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w3_weight: Tensor[(MOE_INTER, DIM), "fp8e4m3"],
    shared_w3_scale: Tensor[(MOE_INTER // 128, DIM // 128), "f8e8m0"],
    shared_w2_weight: Tensor[(DIM, MOE_INTER), "fp8e4m3"],
    shared_w2_scale: Tensor[(DIM // 128, MOE_INTER // 128), "f8e8m0"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    hidden = pre_moe_rms_norm(x, rms_weight)
    routed_value: where(layout=(_, _, H @ cta)) = routed_expert(
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
    shared_value = shared_expert(
        hidden,
        shared_w1_weight,
        shared_w1_scale,
        shared_w3_weight,
        shared_w3_scale,
        shared_w2_weight,
        shared_w2_scale,
    )
    return combine_expert_outputs(routed_value, shared_value)


__all__ = [
    "DIM",
    "MOE_INTER",
    "N_ACT",
    "N_ROUTED",
    "combine_expert_outputs",
    "dsv4_moe_layer",
    "pre_moe_rms_norm",
    "routed_expert",
    "shared_expert",
]
