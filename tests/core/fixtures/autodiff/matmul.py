"""MatMul in each operand layout, and with a batch axis only one operand has."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Matmul:
    @func
    def forward(
        x: Tensor[(2, 4, 3), "f32"], a: Tensor[(3, 5), "f32"], b: Tensor[(6, 5), "f32"], c: Tensor[(6, 4), "f32"]
    ):
        h = tf.matmul(x, a)
        k = tf.matmul(h, b, b_layout="NK")
        y = tf.matmul(c, k, a_layout="KM", b_layout="NK")
        return tf.matmul(y, tf.transpose(y, perm=(0, 2, 1)), a_layout="KM")
