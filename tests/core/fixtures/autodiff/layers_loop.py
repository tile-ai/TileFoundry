"""Three layers in a loop; the body calls a sibling layer with the weights the induction picks."""
from tilefoundry.dsl import *
from tilefoundry.target import cpu

L, T, C, F = 3, 4, 8, 16


@module(entry="forward", target=cpu.Target())
class Layers:
    @func
    def layer(h: Tensor[(T, C), "f32"], wn: Tensor[(C,), "f32"], w1: Tensor[(F, C), "f32"], w2: Tensor[(C, F), "f32"]):
        a = tf.matmul(tf.rms_norm(h, wn, eps=1e-6), w1, b_layout="NK")
        return h + tf.matmul(tf.silu(a), w2, b_layout="NK")

    @func
    def forward(x: Tensor[(T, C), "f32"], wn0: Tensor[(C,), "f32"], norms: Tensor[(L, C), "f32"],
                up: Tensor[(L, F, C), "f32"], down: Tensor[(L, C, F), "f32"]):
        x0 = tf.rms_norm(x, wn0, eps=1e-6)
        h = x0 * tf.exp(x0)
        for t in range(L):
            index = tf.reshape(t, new_shape=(1,))
            wn = tf.reshape(tf.index_select(norms, index, dim=0), new_shape=(C,))
            w1 = tf.reshape(tf.index_select(up, index, dim=0), new_shape=(F, C))
            w2 = tf.reshape(tf.index_select(down, index, dim=0), new_shape=(C, F))
            h = layer(h, wn, w1, w2)
        return h * h
