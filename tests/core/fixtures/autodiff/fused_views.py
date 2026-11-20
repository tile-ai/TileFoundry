"""q, k and v are slices of one fused projection: a saved slice keeps the whole projection."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Fused:
    @func
    def forward(x: Tensor[(8, 16), "f32"], w: Tensor[(16, 48), "f32"]):
        qkv = tf.matmul(x, w)
        q = tf.slice(qkv, (0, 0), sizes=(8, 16), strides=(1, 1))
        k = tf.slice(qkv, (0, 16), sizes=(8, 16), strides=(1, 1))
        v = tf.slice(qkv, (0, 32), sizes=(8, 16), strides=(1, 1))
        qk = q * k
        return qk * v
