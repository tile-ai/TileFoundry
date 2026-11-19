"""One examples/fsdp_training block on a (replicate, shard) mesh of four GPUs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

B, T, C, H, G, D, F = 8, 8, 16, 4, 2, 4, 64
N = B * T
EPS = 1e-6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class HsdpBlock:
    @func(mesh=Mesh(("gpu",), (2, 2), names=("replicate", "shard")))
    def forward(
        x: Tensor[(N @ (mesh.replicate, mesh.shard), C), "f32", "gmem"],
        wq: Tensor[(C @ mesh.shard, C), "f32", "gmem"],
        wk: Tensor[(G * D @ mesh.shard, C), "f32", "gmem"],
        wv: Tensor[(G * D @ mesh.shard, C), "f32", "gmem"],
        wo: Tensor[(C @ mesh.shard, C), "f32", "gmem"],
        wu: Tensor[(F @ mesh.shard, C), "f32", "gmem"],
        wd: Tensor[(C @ mesh.shard, F), "f32", "gmem"],
        rotation: Tensor[(1, 1, 1, T, D, D), "f32", (1, 1, 1, T, D, D) @ {B()}, "gmem"],
        mask: Tensor[(1, 1, 1, T, T), "f32", (1, 1, 1, T, T) @ {B()}, "gmem"],
    ):
        q_full = tf.reshard(wq, layout=(C, C) @ {B()}, storage="gmem")
        k_full = tf.reshard(wk, layout=(G * D, C) @ {B()}, storage="gmem")
        v_full = tf.reshard(wv, layout=(G * D, C) @ {B()}, storage="gmem")
        o_full = tf.reshard(wo, layout=(C, C) @ {B()}, storage="gmem")
        u_full = tf.reshard(wu, layout=(F, C) @ {B()}, storage="gmem")
        d_full = tf.reshard(wd, layout=(C, F) @ {B()}, storage="gmem")
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
