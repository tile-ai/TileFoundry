"""MoE gate: TopK of router logits, softmax of the chosen scores, IndexSelect of expert weights."""
from tilefoundry.dsl import *
from tilefoundry.target import cpu

T, C, E, K, F = 4, 6, 5, 2, 8


@module(entry="forward", target=cpu.Target())
class Moe:
    @func
    def forward(x: Tensor[(T, C), "f32"], wg: Tensor[(E, C), "f32"], experts: Tensor[(E, F, C), "f32"]):
        logits = tf.matmul(x, wg, b_layout="NK")
        scores, chosen = tf.topk(logits, k=K, axis=-1)
        p = tf.softmax(scores, axis=-1)
        w = tf.index_select(experts, tf.reshape(chosen, new_shape=(T * K,)), dim=0)
        xs = tf.reshape(tf.repeat_interleave(x, repeats=K, axis=0), new_shape=(T * K, 1, C))
        h = tf.reshape(tf.matmul(xs, w, b_layout="NK"), new_shape=(T, K, F))
        y = tf.reduce(h * tf.reshape(p, new_shape=(T, K, 1)), axes=(1,), kind=ReduceKind.SUM, keepdim=False)
        return tf.silu(y)
