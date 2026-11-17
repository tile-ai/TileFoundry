"""A recurrence whose step is a function, the shape of a GDN single-step update."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopCall:
    @func
    def step(state: Tensor[(3, 3), "f32"], k: Tensor[(1, 3), "f32"], decay: Tensor[(3, 3), "f32"]):
        update = tf.matmul(k, k, a_layout="KM")
        return state * tf.exp(-decay) + update, tf.matmul(k, state)

    @func
    def forward(keys: Tensor[(4, 3), "f32"], decay: Tensor[(3, 3), "f32"]):
        state = tf.zeros(Tensor[(3, 3), "f32"])
        out = tf.zeros(Tensor[(1, 3), "f32"])
        for t in range(4):
            k = tf.index_select(keys, tf.reshape(t, new_shape=(1,)), dim=0)
            state, o = step(state, k, decay)
            out = out + o * o
        return out
