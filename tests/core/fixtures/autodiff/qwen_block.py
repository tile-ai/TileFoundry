"""A Qwen-like decoder block: RMSNorm, GQA with RoPE and a causal mask, SwiGLU, residuals."""
from tilefoundry.dsl import *
from tilefoundry.target import cpu

T, C, H, KV, D, F = 8, 32, 4, 2, 8, 64


@module(entry="forward", target=cpu.Target())
class Qwen:
    @func
    def forward(
        x: Tensor[(T, C), "f32"], wn1: Tensor[(C,), "f32"], wq: Tensor[(H * D, C), "f32"],
        wk: Tensor[(KV * D, C), "f32"], wv: Tensor[(KV * D, C), "f32"], wo: Tensor[(C, H * D), "f32"],
        wn2: Tensor[(C,), "f32"], wg: Tensor[(F, C), "f32"], wu: Tensor[(F, C), "f32"],
        wd: Tensor[(C, F), "f32"], half_cos: Tensor[(T, D // 2), "f32"], half_sin: Tensor[(T, D // 2), "f32"],
        pos: Tensor[(T,), "i32"], mask: Tensor[(1, 1, T, T), "f32"],
    ):
        h = tf.rms_norm(x, wn1, eps=1e-6)
        q = tf.reshape(tf.matmul(h, wq, b_layout="NK"), new_shape=(1, T, H, D))
        k = tf.reshape(tf.matmul(h, wk, b_layout="NK"), new_shape=(1, T, KV, D))
        v = tf.reshape(tf.matmul(h, wv, b_layout="NK"), new_shape=(1, T, KV, D))
        cos = tf.concat([half_cos, half_cos], axis=-1)
        sin = tf.concat([half_sin, half_sin], axis=-1)
        rq, rk = tf.rope(q, k, cos, sin, pos)
        qh = tf.transpose(rq, perm=(0, 2, 1, 3))
        kh = tf.repeat_interleave(tf.transpose(rk, perm=(0, 2, 1, 3)), repeats=2, axis=1)
        vh = tf.repeat_interleave(tf.transpose(v, perm=(0, 2, 1, 3)), repeats=2, axis=1)
        scores = tf.matmul(qh, kh, b_layout="NK") * 0.35355339 + mask
        p = tf.softmax(scores, axis=-1)
        att = tf.reshape(tf.transpose(tf.matmul(p, vh), perm=(0, 2, 1, 3)), new_shape=(T, H * D))
        x1 = x + tf.matmul(att, wo, b_layout="NK")
        h2 = tf.rms_norm(x1, wn2, eps=1e-6)
        gate = tf.matmul(h2, wg, b_layout="NK")
        up = tf.matmul(h2, wu, b_layout="NK")
        return x1 + tf.matmul(tf.silu(gate) * up, wd, b_layout="NK")
