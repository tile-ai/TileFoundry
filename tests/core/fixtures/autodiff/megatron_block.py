"""Megatron tensor parallel on two GPUs: column-split q/k/v/gate/up, row-split o/down.

RMSNorm and SoftMax read Broadcast values whose gradient a column-split matmul makes Partial.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

T, C, H, D, F = 8, 16, 4, 4, 32


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class TensorParallel:
    @func(mesh=Mesh(("gpu",), (2,), names=("tp",)))
    def forward(
        x: Tensor[(T, C), "f32", (T, C), "gmem"],
        wn1: Tensor[(C,), "f32", (C,), "gmem"],
        wq: Tensor[(H * D @ mesh.tp, C), "f32", "gmem"],
        wk: Tensor[(H * D @ mesh.tp, C), "f32", "gmem"],
        wv: Tensor[(H * D @ mesh.tp, C), "f32", "gmem"],
        wo: Tensor[(C, H * D @ mesh.tp), "f32", "gmem"],
        wn2: Tensor[(C,), "f32", (C,), "gmem"],
        wg: Tensor[(F @ mesh.tp, C), "f32", "gmem"],
        wu: Tensor[(F @ mesh.tp, C), "f32", "gmem"],
        wd: Tensor[(C, F @ mesh.tp), "f32", "gmem"],
        mask: Tensor[(1, T, T), "f32", (1, T, T), "gmem"],
    ):
        h = tf.rms_norm(x, wn1, eps=1e-6)
        q = tf.transpose(tf.reshape(tf.matmul(h, wq, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        k = tf.transpose(tf.reshape(tf.matmul(h, wk, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        v = tf.transpose(tf.reshape(tf.matmul(h, wv, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        p = tf.softmax(tf.matmul(q, k, b_layout="NK") * 0.5 + mask, axis=-1)
        att = tf.reshape(tf.transpose(tf.matmul(p, v), perm=(1, 0, 2)), new_shape=(T, H * D))
        o = tf.reshard(tf.matmul(att, wo, b_layout="NK"), layout=(T, C), storage="gmem")
        x1 = x + o
        h2 = tf.rms_norm(x1, wn2, eps=1e-6)
        m = tf.silu(tf.matmul(h2, wg, b_layout="NK")) * tf.matmul(h2, wu, b_layout="NK")
        d = tf.reshard(tf.matmul(m, wd, b_layout="NK"), layout=(T, C), storage="gmem")
        return x1 + d
