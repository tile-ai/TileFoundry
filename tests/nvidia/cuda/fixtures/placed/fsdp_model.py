"""The examples/fsdp_training model on two H200s: embedding, two GQA blocks, head, cross entropy.

Each block gathers its Split(0) weights to Broadcast; the embedding is a one-hot matmul.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

B, T, C, H, G, D, F, V = 8, 8, 16, 4, 2, 4, 64, 32
N = B * T
EPS = 1e-6
MEAN = 1.0 / N


@module(entry="forward", target=cuda.Target("nvidia.h200_sxm", device_count=2), topologies=(Topology("gpu", 2), Topology("cta", 1)))
class FsdpModel:
    @func(mesh=Mesh(("gpu",), (2,), names=("shard",)))
    def block(
        x: Tensor[(N, C), "f32", (N @ mesh.shard, C), "gmem"],
        wq: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        wk: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        wv: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        wo: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        wu: Tensor[(F, C), "f32", (F @ mesh.shard, C), "gmem"],
        wd: Tensor[(C, F), "f32", (C @ mesh.shard, F), "gmem"],
        rotation: Tensor[(1, 1, 1, T, D, D), "f32", (1, 1, 1, T, D, D), "gmem"],
        mask: Tensor[(1, 1, 1, T, T), "f32", (1, 1, 1, T, T), "gmem"],
    ):
        q_full = tf.reshard(wq, layout=(C, C), storage="gmem")
        k_full = tf.reshard(wk, layout=(G * D, C), storage="gmem")
        v_full = tf.reshard(wv, layout=(G * D, C), storage="gmem")
        o_full = tf.reshard(wo, layout=(C, C), storage="gmem")
        u_full = tf.reshard(wu, layout=(F, C), storage="gmem")
        d_full = tf.reshard(wd, layout=(C, F), storage="gmem")
        ix = tf.rsqrt(tf.reduce(x * x, axes=(-1,)) + EPS)
        a = x * ix
        q0 = tf.transpose(tf.reshape(tf.matmul(a, q_full, b_layout="NK"), (B, T, G, H // G, D)), perm=(0, 2, 3, 1, 4))
        k0 = tf.transpose(tf.reshape(tf.matmul(a, k_full, b_layout="NK"), (B, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
        v = tf.transpose(tf.reshape(tf.matmul(a, v_full, b_layout="NK"), (B, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
        qn = q0 * tf.rsqrt(tf.reduce(q0 * q0, axes=(-1,)) + EPS)
        kn = k0 * tf.rsqrt(tf.reduce(k0 * k0, axes=(-1,)) + EPS)
        q = tf.reshape(tf.matmul(tf.reshape(qn, (B, G, H // G, T, 1, D)), rotation), (B, G, H // G, T, D))
        k = tf.reshape(tf.matmul(tf.reshape(kn, (B, G, 1, T, 1, D)), rotation), (B, G, 1, T, D))
        probability = tf.softmax(tf.matmul(q, k, b_layout="NK") * 0.5 + mask, axis=-1)
        attended = tf.reshape(tf.transpose(tf.matmul(probability, v), perm=(0, 3, 1, 2, 4)), (N, C))
        residual = x + tf.matmul(attended, o_full, b_layout="NK")
        r = residual * tf.rsqrt(tf.reduce(residual * residual, axes=(-1,)) + EPS)
        relu = tf.relu(tf.matmul(r, u_full, b_layout="NK"))
        return residual + tf.matmul(relu * relu, d_full, b_layout="NK")

    @func(mesh=Mesh(("gpu",), (2,), names=("shard",)))
    def forward(
        tokens: Tensor[(N, V), "f32", (N @ mesh.shard, V), "gmem"],
        targets: Tensor[(N, V), "f32", (N @ mesh.shard, V), "gmem"],
        rotation: Tensor[(1, 1, 1, T, D, D), "f32", (1, 1, 1, T, D, D), "gmem"],
        mask: Tensor[(1, 1, 1, T, T), "f32", (1, 1, 1, T, T), "gmem"],
        we: Tensor[(V, C), "f32", (V @ mesh.shard, C), "gmem"],
        q1: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        k1: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        v1: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        o1: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        u1: Tensor[(F, C), "f32", (F @ mesh.shard, C), "gmem"],
        d1: Tensor[(C, F), "f32", (C @ mesh.shard, F), "gmem"],
        q2: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        k2: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        v2: Tensor[(G * D, C), "f32", (G * D @ mesh.shard, C), "gmem"],
        o2: Tensor[(C, C), "f32", (C @ mesh.shard, C), "gmem"],
        u2: Tensor[(F, C), "f32", (F @ mesh.shard, C), "gmem"],
        d2: Tensor[(C, F), "f32", (C @ mesh.shard, F), "gmem"],
        wh: Tensor[(V, C), "f32", (V @ mesh.shard, C), "gmem"],
    ):
        x = tf.matmul(tokens, tf.reshard(we, layout=(V, C), storage="gmem"))
        y1 = block(x, q1, k1, v1, o1, u1, d1, rotation, mask)
        y2 = block(y1, q2, k2, v2, o2, u2, d2, rotation, mask)
        z = y2 * tf.rsqrt(tf.reduce(y2 * y2, axes=(-1,)) + EPS)
        logits = tf.matmul(z, tf.reshard(wh, layout=(V, C), storage="gmem"), b_layout="NK")
        shifted = logits - tf.reduce(logits, axes=(-1,), kind=ReduceKind.MAX)
        log_z = tf.log(tf.reduce(tf.exp(shifted), axes=(-1,), kind=ReduceKind.SUM))
        nll = -tf.reduce(targets * (shifted - log_z), axes=(-1,), kind=ReduceKind.SUM)
        return tf.reduce(nll * MEAN, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
