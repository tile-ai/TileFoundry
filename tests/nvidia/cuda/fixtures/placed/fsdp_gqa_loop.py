"""Two FSDP attention layers in a loop on two H200s; repeat_interleave expands each K/V head.

The tokens are split over the GPUs and K/V are gathered to Broadcast, so the
heads that repeat_interleave expands are the same on both GPUs, and so is their
gradient once it is reduced.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

L, T, C, H, KV, D = 2, 8, 16, 2, 1, 8


@module(entry="forward", target=cuda.Target("nvidia.h200_sxm", device_count=2), topologies=(Topology("gpu", 2), Topology("cta", 1)))
class FsdpGqaLoop:
    @func(mesh=Mesh(("gpu",), (2,), names=("cp",)))
    def layer(
        x: Tensor[(T @ mesh.cp, C), "f32", "gmem"],
        wq: Tensor[(H * D, C), "f32", (H * D, C), "gmem"],
        wk: Tensor[(KV * D, C), "f32", (KV * D, C), "gmem"],
        wv: Tensor[(KV * D, C), "f32", (KV * D, C), "gmem"],
        wo: Tensor[(C, H * D), "f32", (C, H * D), "gmem"],
        mask: Tensor[(1, 1, T @ mesh.cp, T), "f32", "gmem"],
    ):
        q = tf.transpose(tf.reshape(tf.matmul(x, wq, b_layout="NK"), new_shape=(1, T, H, D)), perm=(0, 2, 1, 3))
        k = tf.reshape(tf.matmul(x, wk, b_layout="NK"), new_shape=(1, T, KV, D))
        v = tf.reshape(tf.matmul(x, wv, b_layout="NK"), new_shape=(1, T, KV, D))
        k_all = tf.reshard(k, layout=(1, T, KV, D), storage="gmem")
        v_all = tf.reshard(v, layout=(1, T, KV, D), storage="gmem")
        kh = tf.repeat_interleave(tf.transpose(k_all, perm=(0, 2, 1, 3)), repeats=H // KV, axis=1)
        vh = tf.repeat_interleave(tf.transpose(v_all, perm=(0, 2, 1, 3)), repeats=H // KV, axis=1)
        p = tf.softmax(tf.matmul(q, kh, b_layout="NK") * 0.35355339 + mask, axis=-1)
        att = tf.reshape(tf.transpose(tf.matmul(p, vh), perm=(0, 2, 1, 3)), new_shape=(T, H * D))
        return x + tf.matmul(att, wo, b_layout="NK")

    @func(mesh=Mesh(("gpu",), (2,), names=("cp",)))
    def forward(
        x: Tensor[(T @ mesh.cp, C), "f32", "gmem"],
        wq: Tensor[(L, H * D @ mesh.cp, C), "f32", "gmem"],
        wk: Tensor[(L, KV * D @ mesh.cp, C), "f32", "gmem"],
        wv: Tensor[(L, KV * D @ mesh.cp, C), "f32", "gmem"],
        wo: Tensor[(L, C @ mesh.cp, H * D), "f32", "gmem"],
        mask: Tensor[(1, 1, T @ mesh.cp, T), "f32", "gmem"],
    ):
        h = x
        for t in range(L):
            i = tf.reshape(t, new_shape=(1,))
            fq = tf.reshard(tf.reshape(tf.index_select(wq, i, dim=0), new_shape=(H * D, C)), layout=(H * D, C), storage="gmem")
            fk = tf.reshard(tf.reshape(tf.index_select(wk, i, dim=0), new_shape=(KV * D, C)), layout=(KV * D, C), storage="gmem")
            fv = tf.reshard(tf.reshape(tf.index_select(wv, i, dim=0), new_shape=(KV * D, C)), layout=(KV * D, C), storage="gmem")
            fo = tf.reshard(tf.reshape(tf.index_select(wo, i, dim=0), new_shape=(C, H * D)), layout=(C, H * D), storage="gmem")
            h = layer(h, fq, fk, fv, fo, mask)
        return tf.reduce(h * h, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
